"""
Climate Explorer — FastAPI Backend
Serves HTML templates and provides CDS API integration for fetching
real climate data from Copernicus ERA5.
"""

import os
import uuid
import math
import csv
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import xarray as xr
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Climate Explorer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Directories ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
DOWNLOADS_DIR = BASE_DIR / "downloads"
DOWNLOADS_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

# Serve templates as static HTML
app.mount("/static", StaticFiles(directory=str(TEMPLATES_DIR)), name="static")

# ── CDS config ───────────────────────────────────────────────
CDS_API_URL = os.getenv("CDS_API_URL", "https://cds.climate.copernicus.eu/api")
CDS_API_KEY = os.getenv("CDS_API_KEY", "")

# ── Variable metadata for unit conversion ────────────────────
# Maps CDS variable names to human-friendly column names + conversion
VARIABLE_INFO = {
    "2m_temperature":           {"col": "T2M",   "unit": "°C",   "convert": lambda v: v - 273.15},
    "2m_dewpoint_temperature":  {"col": "DEWP",  "unit": "°C",   "convert": lambda v: v - 273.15},
    "skin_temperature":         {"col": "SKT",   "unit": "°C",   "convert": lambda v: v - 273.15},
    "sea_surface_temperature":  {"col": "SST",   "unit": "°C",   "convert": lambda v: v - 273.15},
    "surface_pressure":         {"col": "SP",    "unit": "hPa",  "convert": lambda v: v / 100.0},
    "mean_sea_level_pressure":  {"col": "MSLP",  "unit": "hPa",  "convert": lambda v: v / 100.0},
    "total_precipitation":      {"col": "PRCP",  "unit": "mm",   "convert": lambda v: v * 1000.0},
    "10m_u_component_of_wind":  {"col": "U10",   "unit": "m/s",  "convert": None},
    "10m_v_component_of_wind":  {"col": "V10",   "unit": "m/s",  "convert": None},
    "100m_u_component_of_wind": {"col": "U100",  "unit": "m/s",  "convert": None},
    "100m_v_component_of_wind": {"col": "V100",  "unit": "m/s",  "convert": None},
    "10m_wind_gust_since_previous_post_processing": {"col": "GUST", "unit": "m/s", "convert": None},
    "surface_solar_radiation_downwards":   {"col": "SSRD",  "unit": "J/m²", "convert": None},
    "surface_thermal_radiation_downwards": {"col": "STRD",  "unit": "J/m²", "convert": None},
    "mean_wave_direction":      {"col": "MWD",   "unit": "°",    "convert": None},
    "mean_wave_period":         {"col": "MWP",   "unit": "s",    "convert": None},
    "significant_height_of_combined_wind_waves_and_swell": {"col": "SWH", "unit": "m", "convert": None},
}

# Map CDS variable names to the actual NetCDF short-name keys
CDS_TO_NC_VAR = {
    "2m_temperature":           "t2m",
    "2m_dewpoint_temperature":  "d2m",
    "skin_temperature":         "skt",
    "sea_surface_temperature":  "sst",
    "surface_pressure":         "sp",
    "mean_sea_level_pressure":  "msl",
    "total_precipitation":      "tp",
    "10m_u_component_of_wind":  "u10",
    "10m_v_component_of_wind":  "v10",
    "100m_u_component_of_wind": "u100",
    "100m_v_component_of_wind": "v100",
    "10m_wind_gust_since_previous_post_processing": "i10fg",
    "surface_solar_radiation_downwards":   "ssrd",
    "surface_thermal_radiation_downwards": "strd",
    "mean_wave_direction":      "mwd",
    "mean_wave_period":         "mwp",
    "significant_height_of_combined_wind_waves_and_swell": "swh",
}


# ── Request model ────────────────────────────────────────────
class CDSRequest(BaseModel):
    variables: List[str]        # e.g. ["2m_temperature", "total_precipitation"]
    latitude: float
    longitude: float
    start_date: str             # "YYYY-MM-DD"
    end_date: str               # "YYYY-MM-DD"


def smart_open_dataset(path: str):
    """Open a dataset, auto-detecting ZIP/NetCDF/GRIB format."""
    import zipfile

    # Check magic bytes to determine format
    with open(path, 'rb') as f:
        magic = f.read(8)

    # ZIP files start with 'PK\x03\x04' — CDS API often returns zipped NetCDF
    if magic[:4] == b'PK\x03\x04':
        extract_dir = os.path.join(os.path.dirname(path), '_extracted')
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(path, 'r') as zf:
            nc_files = [n for n in zf.namelist() if n.endswith('.nc')]
            data_files = nc_files if nc_files else zf.namelist()
            zf.extractall(extract_dir, members=data_files)
        # Open all extracted NC files and merge them
        datasets = []
        for fname in data_files:
            extracted_path = os.path.join(extract_dir, fname)
            try:
                ds = xr.open_dataset(extracted_path, engine='netcdf4')
                datasets.append(ds)
            except Exception:
                pass
        if not datasets:
            raise RuntimeError(f"ZIP contained no readable data files: {data_files}")
        if len(datasets) == 1:
            return datasets[0]
        return xr.merge(datasets)

    # NetCDF files start with 'CDF' or '\x89HDF'
    if magic[:3] == b'CDF' or magic[:4] == b'\x89HDF':
        return xr.open_dataset(path, engine='netcdf4')

    # GRIB files start with 'GRIB'
    if magic[:4] == b'GRIB':
        try:
            import cfgrib
            datasets = cfgrib.open_datasets(path)
            if len(datasets) == 1:
                return datasets[0]
            return xr.merge(datasets)
        except Exception as e:
            raise RuntimeError(f"Failed to read GRIB file: {e}")

    # Fallback: try netcdf4, then cfgrib
    try:
        return xr.open_dataset(path, engine='netcdf4')
    except Exception:
        try:
            import cfgrib
            datasets = cfgrib.open_datasets(path)
            if len(datasets) == 1:
                return datasets[0]
            return xr.merge(datasets)
        except Exception as e2:
            raise RuntimeError(f"Could not open file as NetCDF or GRIB: {e2}")


def nc_to_cn(nc_path: str, variables: List[str]) -> dict:
    """Convert a NetCDF/GRIB file to .cn (CSV) format and return JSON summary."""
    ds = smart_open_dataset(nc_path)

    # Build time axis
    if "valid_time" in ds.dims:
        time_key = "valid_time"
    elif "time" in ds.dims:
        time_key = "time"
    else:
        time_key = list(ds.dims.keys())[0]

    times = ds[time_key].values
    n = len(times)

    # Build columns and extract data
    columns = ["DATE"]
    data_arrays = {}

    for cds_var in variables:
        nc_var = CDS_TO_NC_VAR.get(cds_var)
        info = VARIABLE_INFO.get(cds_var)
        if not nc_var or not info:
            continue
        if nc_var not in ds.data_vars:
            # Try the CDS long name as fallback
            found = False
            for dv in ds.data_vars:
                if dv.lower() == nc_var.lower():
                    nc_var = dv
                    found = True
                    break
            if not found:
                continue

        col_name = info["col"]
        columns.append(col_name)
        raw = ds[nc_var].values.flatten()[:n]
        if info["convert"]:
            raw = info["convert"](raw.astype(float))
        data_arrays[col_name] = raw

    ds.close()

    # Build records
    records = []
    for i in range(n):
        t = times[i]
        # Convert numpy datetime64 to string
        if hasattr(t, "astype"):
            dt = str(np.datetime_as_string(t, unit="h"))
        else:
            dt = str(t)
        row = {"DATE": dt}
        for col in columns[1:]:
            val = float(data_arrays[col][i])
            if math.isnan(val) or math.isinf(val):
                row[col] = None
            else:
                row[col] = round(val, 4)
        records.append(row)

    # Stats
    stats = {}
    for col in columns[1:]:
        vals = [r[col] for r in records if r[col] is not None]
        if vals:
            stats[col] = {
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "mean": round(sum(vals) / len(vals), 4),
                "count": len(vals),
            }

    # Write .cn file
    cn_name = f"climate_{uuid.uuid4().hex[:8]}.cn"
    cn_path = DOWNLOADS_DIR / cn_name
    with open(cn_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    return {
        "columns": columns,
        "records": records,
        "stats": stats,
        "row_count": len(records),
        "parse_errors": [],
        "filename": cn_name,
        "cn_download": f"/download-cn/{cn_name}",
        "_src": "cds",
    }

def nc_to_cn_generic(nc_path: str) -> dict:
    """Generic NetCDF to .cn conversion — uses variable names as-is."""
    ds = smart_open_dataset(nc_path)

    # Find time dimension
    time_key = None
    for k in ["valid_time", "time"]:
        if k in ds.dims:
            time_key = k
            break
    if not time_key:
        time_key = list(ds.dims.keys())[0] if ds.dims else None

    times = ds[time_key].values if time_key else np.arange(len(list(ds.data_vars.values())[0]))
    n = len(times)

    columns = ["DATE"]
    data_arrays = {}

    for var_name in ds.data_vars:
        col = var_name.upper()
        columns.append(col)
        raw = ds[var_name].values.flatten()[:n].astype(float)
        data_arrays[col] = raw

    ds.close()

    records = []
    for i in range(n):
        if time_key and hasattr(times[i], "astype"):
            dt = str(np.datetime_as_string(times[i], unit="h"))
        else:
            dt = str(times[i])
        row = {"DATE": dt}
        for col in columns[1:]:
            val = float(data_arrays[col][i])
            row[col] = None if (math.isnan(val) or math.isinf(val)) else round(val, 4)
        records.append(row)

    stats = {}
    for col in columns[1:]:
        vals = [r[col] for r in records if r[col] is not None]
        if vals:
            stats[col] = {"min": round(min(vals), 4), "max": round(max(vals), 4),
                          "mean": round(sum(vals) / len(vals), 4), "count": len(vals)}

    cn_name = f"climate_{uuid.uuid4().hex[:8]}.cn"
    cn_path = DOWNLOADS_DIR / cn_name
    with open(cn_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    return {"columns": columns, "records": records, "stats": stats, "row_count": len(records),
            "parse_errors": [], "filename": cn_name, "cn_download": f"/download-cn/{cn_name}", "_src": "file"}



@app.post("/cds-fetch")
async def cds_fetch(req: CDSRequest):
    """Fetch climate data from Copernicus CDS and return as .cn JSON."""
    if not CDS_API_KEY or CDS_API_KEY == "YOUR_CDS_API_KEY_HERE":
        raise HTTPException(
            status_code=500,
            detail="CDS API key not configured. Set CDS_API_KEY in .env file."
        )

    if not req.variables:
        raise HTTPException(status_code=400, detail="Select at least one variable.")

    # Validate dates
    try:
        start = datetime.strptime(req.start_date, "%Y-%m-%d")
        end = datetime.strptime(req.end_date, "%Y-%m-%d")
        if end < start:
            raise ValueError()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date range.")

    # Build CDS API request using the correct ERA5 format
    import cdsapi

    try:
        client = cdsapi.Client(url=CDS_API_URL, key=CDS_API_KEY)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CDS client init failed: {e}")

    # Collect unique years, months, days from the date range
    years = sorted(set(str(y) for y in range(start.year, end.year + 1)))
    months = sorted(set(f"{m:02d}" for m in range(1, 13)
                        if any(datetime(y, m, 1) >= start.replace(day=1)
                               and datetime(y, m, 1) <= end.replace(day=1)
                               for y in range(start.year, end.year + 1))))
    days = sorted(set(f"{d:02d}" for d in range(1, 32)))
    # Use all hours for hourly data, or just 12:00 for daily
    time_steps = ["12:00"]

    # Area: [north, west, south, east] — small box around the point
    lat, lon = req.latitude, req.longitude
    area = [lat + 0.25, lon - 0.25, lat - 0.25, lon + 0.25]

    cds_request = {
        "product_type": ["reanalysis"],
        "variable": req.variables,
        "year": years,
        "month": months,
        "day": days,
        "time": time_steps,
        "area": area,
        "data_format": "netcdf",
    }

    # Download to temp file
    tmp_nc = tempfile.NamedTemporaryFile(suffix=".nc", delete=False, dir=str(DOWNLOADS_DIR))
    tmp_nc.close()

    try:
        client.retrieve(
            "reanalysis-era5-single-levels",
            cds_request,
            tmp_nc.name,
        )
    except Exception as e:
        try:
            os.unlink(tmp_nc.name)
        except OSError:
            pass
        raise HTTPException(status_code=502, detail=f"CDS API error: {e}")

    # Convert NetCDF → .cn JSON
    try:
        data = nc_to_cn(tmp_nc.name, req.variables)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NetCDF conversion error: {e}")
    finally:
        try:
            os.unlink(tmp_nc.name)
        except OSError:
            pass

    return JSONResponse(content=data)


@app.get("/download-cn/{filename}")
async def download_cn(filename: str):
    """Download a generated .cn file."""
    path = DOWNLOADS_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(
        path=str(path),
        filename=filename,
        media_type="text/csv",
    )


@app.get("/list-downloads")
async def list_downloads():
    """List all .cn files in the downloads folder with metadata."""
    files = []
    for f in sorted(DOWNLOADS_DIR.glob("*.cn"), key=lambda p: p.stat().st_mtime, reverse=True):
        # Quick peek at headers and row count
        try:
            with open(f, "r") as fh:
                lines = fh.readlines()
            header = lines[0].strip() if lines else ""
            cols = header.split(",") if header else []
            row_count = len(lines) - 1 if len(lines) > 1 else 0
            # Check if file has actual numeric data (not all empty)
            has_data = any(
                any(c.strip() and c.strip().replace('.', '').replace('-', '').isdigit()
                    for c in line.split(",")[1:])
                for line in lines[1:4]
            ) if row_count > 0 else False
            files.append({
                "filename": f.name,
                "columns": cols,
                "row_count": row_count,
                "size_kb": round(f.stat().st_size / 1024, 1),
                "has_data": has_data,
                "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            })
        except Exception:
            pass
    return {"files": files}


@app.get("/load-download/{filename}")
async def load_download(filename: str):
    """Parse an existing .cn file and return JSON data for visualization."""
    if not filename.endswith(".cn"):
        raise HTTPException(status_code=400, detail="Only .cn files supported.")
    path = DOWNLOADS_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    with open(path, "r") as fh:
        text = fh.read()
    lines = text.strip().splitlines()
    if len(lines) < 2:
        raise HTTPException(status_code=400, detail="File is empty or has no data rows.")

    reader = csv.DictReader(lines)
    columns = reader.fieldnames or []
    records = []
    for row in reader:
        rec = {}
        for k, v in row.items():
            try:
                rec[k] = float(v) if v and v.strip() else None
            except (ValueError, TypeError):
                rec[k] = v
        records.append(rec)

    stats = {}
    for col in columns:
        vals = [r.get(col) for r in records
                if isinstance(r.get(col), (int, float)) and r.get(col) is not None
                and not math.isnan(r.get(col))]
        if vals:
            stats[col] = {
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "mean": round(sum(vals) / len(vals), 4),
                "count": len(vals),
            }

    return {
        "columns": columns,
        "records": records,
        "stats": stats,
        "row_count": len(records),
        "parse_errors": [],
        "filename": filename,
        "_src": "file",
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a .cn or .nc file and parse it."""
    fname = file.filename or ""

    # ── Handle .nc (NetCDF) files ──
    if fname.endswith(".nc"):
        content = await file.read()
        tmp_path = DOWNLOADS_DIR / f"upload_{uuid.uuid4().hex[:8]}.nc"
        with open(tmp_path, "wb") as f:
            f.write(content)
        try:
            # Auto-detect variables in the NetCDF
            ds = smart_open_dataset(str(tmp_path))
            nc_vars_found = list(ds.data_vars)
            ds.close()
            # Map nc short names back to CDS variable names
            nc_to_cds = {v: k for k, v in CDS_TO_NC_VAR.items()}
            cds_vars = [nc_to_cds[v] for v in nc_vars_found if v in nc_to_cds]
            # If no known vars, use all data vars as-is
            if cds_vars:
                data = nc_to_cn(str(tmp_path), cds_vars)
            else:
                # Generic NC parsing — use variable names directly
                data = nc_to_cn_generic(str(tmp_path))
            data["filename"] = fname
            data["_src"] = "file"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"NetCDF parse error: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return data

    # ── Handle .cn (CSV) files ──
    if not fname.endswith(".cn"):
        raise HTTPException(status_code=400, detail="Only .cn or .nc files are accepted.")

    content = await file.read()
    text = content.decode("utf-8", errors="replace")
    lines = text.strip().splitlines()

    if len(lines) < 2:
        raise HTTPException(status_code=400, detail="File is empty or has no data rows.")

    reader = csv.DictReader(lines)
    columns = reader.fieldnames or []
    records = []
    parse_errors = []

    for i, row in enumerate(reader):
        rec = {}
        for k, v in row.items():
            try:
                rec[k] = float(v)
            except (ValueError, TypeError):
                rec[k] = v
        records.append(rec)

    # Stats for numeric columns
    stats = {}
    for col in columns:
        vals = []
        for r in records:
            v = r.get(col)
            if isinstance(v, (int, float)) and not math.isnan(v):
                vals.append(v)
        if vals:
            stats[col] = {
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "mean": round(sum(vals) / len(vals), 4),
                "count": len(vals),
            }

    return {
        "columns": columns,
        "records": records,
        "stats": stats,
        "row_count": len(records),
        "parse_errors": parse_errors,
        "filename": file.filename,
    }


# ── Serve HTML pages ─────────────────────────────────────────
from fastapi.responses import HTMLResponse

@app.get("/{page}.html")
async def serve_page(page: str):
    # index.html → serve home.html (nav links use index.html)
    if page == "index":
        page = "home"
    file_path = TEMPLATES_DIR / f"{page}.html"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Page not found.")
    return HTMLResponse(content=file_path.read_text(encoding="utf-8"))

@app.get("/")
async def index():
    file_path = TEMPLATES_DIR / "home.html"
    if file_path.exists():
        return HTMLResponse(content=file_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Climate Explorer</h1><p><a href='/visualize.html'>Visualize</a></p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
