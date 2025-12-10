# 🏙️ UrbanShift DC API

**UrbanShift DC API** is a transparent, data-driven system that quantifies
the "uplift potential" of neighborhoods across Washington, DC — based on
crime, property values, and amenity accessibility. Built with Python,
Pandas, and FastAPI (coming soon).

**Objective**: Compute a transparent “Uplift Potential Score” for each census tract in Washington, DC, based on crime, home-value trends, and amenity access.

**Data Sources**: List links + years (2019-2024) for incidents, arrests, home values, grocery stores, metro.

**Methodology**: Brief description of features and scoring formula (we’ll copy the one we defined).

**Usage**: How to run notebooks, generate maps, interpret results.

---

## Project Structure

```bash
urbanshift-dc-api/
│
├── data/
│ ├── raw/
│ ├── processed/
│ └── metadata/
│
├── notebooks/
│ └── 01_data_prep.ipynb
│
├── src/
│ ├── **init**.py
│ ├── features.py
│ ├── scoring.py
│ ├── api.py
│ └── utils.py
│
├── requirements.txt
├── README.md
└── Dockerfile
```

---

**Next Steps**: Model version (TensorFlow) etc.
