# 🏙️ UrbanShift DC API

**Data-Driven Uplift Potential Scoring for Washington, DC Neighborhoods**

UrbanShift DC API is an interpretable machine learning system that estimates the Uplift Potential Score of Washington, DC census tracts — a metric designed to highlight areas with future revitalization or gentrification potential based on crime patterns, accessibility, and housing indicators.

This project demonstrates:

- Real-world data ingestion (incidents, arrests, population)

- Feature engineering for social & urban analytics

- A TensorFlow regression model for uplift scoring

- A production-ready FastAPI inference service

- Clean environment variable handling with .env

- Docker-ready structure (full containerization coming next)\*

---

## 🌐 Objective

**Compute a transparent, explainable uplift score for each census tract, based on:**

- Violent incidents

- Drug-related arrests

- Population-normalized crime rate

- Amenity accessibility

- Relative home value index

This score helps identify neighborhoods that may be positioned for long-term improvement or investment.

---

## Project Structure

```bash
urbanshift-dc-api/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── notebooks/
│   └── 01_data_prep.ipynb
│
├── src/
│   ├── __init__.py
│   ├── features.py
│   ├── scoring.py
│   ├── predict.py
│   └── utils.py
│
├── models/
│   └── uplift_model.keras
│
├── app.py
├── requirements.txt
├── .env
└── README.md
```

---

## 📊 Data Sources (2019–2024)

**Crime & Arrests**

- DC Crime Incidents
  https://opendata.dc.gov/datasets/crime-incidents/

- Adult Arrests
  https://opendata.dc.gov/datasets/adult-arrests/

**Census & Demographics**

- Census ACS Population by Tract
  https://api.census.gov/

**Geographic Data**

- DC Census Tracts (Shapefiles)
  https://www2.census.gov/geo/tiger/

(Additional sources such as home values, grocery store access, and metro proximity will be layered in later versions.)

---

## 🧮 Methodology

### Step 1 — Feature Engineering

| **Feature**         | **Description**                                                   |
| ------------------- | ----------------------------------------------------------------- |
| crime_rate_per_1000 | (violent_incidents + drug_arrests) normalized by tract population |
| accessibility_score | Manually defined or derived from transportation/amenities         |
| home_value_score    | Scaled housing index (0–1)                                        |
| total_crime         | Combined violent + drug arrest activity                           |

### Step 2 — Uplift Score Formula (current version)

A simple, transparent model:

```text
uplift = 0.4 * (1 - normalized_crime_rate)
       + 0.3 * accessibility_score
       + 0.3 * home_value_score
```

### Step 3 — Machine Learning Model

A lightweight **TensorFlow regression model** trains on engineered features:

- Input: [crime_rate_per_1000, accessibility_score, home_value_score]

- Output: uplift_score in range (0,1)

Model is saved as:

```bash
models/uplift_model.keras
```

---

## ⚙️ FastAPI Inference Service

The service loads the TensorFlow model once at startup and exposes:

`GET /health`

Check service health.

`POST /predict`

Run uplift prediction from raw parameters.

**Request Body**

```json
{
  "crime_count": 25,
  "population": 4200,
  "accessibility_score": 0.7,
  "home_value_score": 0.4
}
```

**Response**

```json
{
  "uplift_score": 0.2976,
  "crime_rate_per_1000": 5.952
}
```

---

## 🛠️ Running Locally (Dev)

1. Create .env file

```text
CENSUS_API_KEY=YOUR_KEY_HERE
```

.env is **gitignored** for security.

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Start FastAPI

```bash
uvicorn app:app --reload
```

4. Open API Docs

http://127.0.0.1:8000/docs

---

## 🐳 Docker

We will create a full production-ready image tomorrow.

Planned structure:

```bash
docker build -t urbanshift-api .
docker run -p 8000:8000 --env-file .env urbanshift-api
```

---

## 👨‍💻 Author

### Kevin Woods

Applied ML Engineer

AWS Certified AI Practitioner

AWS Machine Learning Certified Engineer – Associate

- 🔗 [GitHub: woodskevinj](https://github.com/woodskevinj)
