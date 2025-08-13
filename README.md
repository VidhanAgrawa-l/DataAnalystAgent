# Data Analyst Agent

A FastAPI-based data analysis agent that can:
- Scrape Wikipedia data and analyze highest-grossing films
- Analyze Indian High Court judgment data using DuckDB
- Process weather data from APIs
- Handle CSV file analysis
- Generate visualizations and return them as base64-encoded images

## Features

- **Wikipedia Analysis**: Scrapes film data and performs statistical analysis
- **Court Data Analysis**: Uses DuckDB to query large datasets on S3
- **Weather Data**: Fetches current weather from APIs
- **CSV Processing**: Analyzes uploaded CSV files
- **Visualization**: Creates charts and plots as base64 data URIs

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

API will be available at `http://localhost:8000/api/`

### Example Request

```bash
curl -X POST "http://localhost:8000/api/" -F "files=@questions.txt"
```

## API Specification

- **Endpoint**: `POST /api/`
- **Input**: Multipart form data with `questions.txt` file and optional additional files
- **Output**: JSON array or JSON object depending on question format
- **Timeout**: Responds within 5 minutes

## Response Formats

### JSON Array (for Wikipedia, CSV analysis)
```json
[1, "Titanic", -0.485782, "data:image/png;base64,iVBORw0KG..."]
```

### JSON Object (for Court data, Weather)
```json
{
  "Which high court disposed the most cases from 2019 - 2022?": "33_10",
  "What's the regression slope...": -1.234567,
  "Plot the year and # of days...": "data:image/webp;base64,..."
}
```

## Deployment

Deployed using ngrok for temporary public access.

## License

MIT License