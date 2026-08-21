# Climate Explorer

Climate Explorer is a dashboard application designed to visualize and explore historical climate data. It allows users to ingest climate data (from synthetic generation, local files, or directly via the Copernicus Climate Data Store API), analyze it through interactive Plotly charts, view geographic distribution on a map or 3D globe, and compare multiple datasets side-by-side.

## Features

- **Dashboard**: Features a centralized view with a 3D PyDeck globe, a 2D spatial Plotly map, and high-level data summaries.
- **Visualize**: An interactive charting tool using Plotly to plot variables like Temperature, Precipitation, Humidity, etc., as line charts, bar charts, scatter plots, heatmaps, box plots, and histograms.
- **Compare**: Generate or load two separate datasets to statistically and visually compare their metrics.
- **Story Mode**: A guided tour highlighting significant historical climate anomalies with interactive visual representations.
- **Data Integration**: 
  - **Manual Generation**: Synthesize data based on climate zones and latitude/longitude.
  - **File Upload**: Upload `.cn` (or supported `.nc`) files for visualization.
  - **CDS API**: Fetch real ERA5 climate data directly from the Copernicus Climate Data Store.

## Prerequisites

- **Python 3.8+**
- (Optional but recommended) A virtual environment.

## Installation

1. **Clone the repository** (or download the source code).
2. **Install the required Python packages**:
   ```bash
   pip install -r requirements.txt
   ```
   *Key dependencies include FastAPI, Uvicorn, cdsapi, xarray, netCDF4, and pydeck.*

3. **Configure API Keys (Optional but required for real data fetching)**:
   Create a `.env` file in the root directory and add your CDS API credentials if you want to fetch live data from Copernicus:
   ```env
   CDS_API_URL=https://cds.climate.copernicus.eu/api
   CDS_API_KEY=your-personal-api-key
   ```
   *You can obtain a key by registering at the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/).*

## Running the Application

Start the FastAPI backend server using Uvicorn:

```bash
uvicorn main:app --reload --port 8000
```

Once the server is running, open your web browser and navigate to:
**http://localhost:8000**

## Project Structure

- `main.py`: The FastAPI application, handling API integrations, file uploads, data conversion, and serving HTML templates.
- `requirements.txt`: Python dependencies.
- `templates/`: Contains all the raw HTML files for the frontend views.
  - `index.html`: Landing page.
  - `dashboard.html`: The main dashboard integrating PyDeck and Plotly.
  - `visualize.html`: Detailed charting interface.
  - `compare.html`: Dual-dataset comparison tool.
  - `story.html`: Guided tour of climate anomalies.

## Data Sources

- **Copernicus ERA5**: Real historical climate data can be fetched via the UI using the "Fetch from CDS API" feature. This communicates with `cdsapi` on the backend.
- **Sample Files**: You can generate synthetic `.cn` sample files directly in the `compare.html` or `visualize.html` pages by using the "Manual" generation feature, which simulates data based on selected climate zones. You can then download these `.cn` files to use later.
