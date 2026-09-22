# City Mobility & Accessibility Score
Scoring a city on how easy it is to navigate and how accessible the city is for a disabled traveler. We use amenity density, accessibility tags and street-level mobility data pulled live form OpenStreetMap (via Geoapify). Build a model that predicts this score from a secondary infrastructure alone, and clustered cities by infrastructure type.

## Links
[Live Web App](https://mobility-index-vtmfvggppevqhkkkp3yibd.streamlit.app/)

## Objective
When someone with a disability is planning to travel there is no one way to find out how accessible the city can be before actually visiting it. This project aims to scores cities on accessibility (How easy it is find basic amenities like healthcare, essentials, hotels, restaurant and how accessible are they) and mobility(How easy it is navigate and walk around the city with shops, bench, parking availability, street smoothness).

## Tech Stack & Tools
- **Python Language**: `python 3.14`
- **Data Collection**: Geoapify Places API, OpenStreetMap data, Wikidata SPARQL (for population)
- **Modeling**: `pandas`, `numpy`, `scikit-learn`, `shap`
- **Deployment**: `streamlit`

## Repository Structure
```
city-mobility-score/
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   └── 02_modeling.ipynb
├── models/
├── data/
├── app.py
├── requirements.txt
└── README.md
```

## Pipeline

### Data Preparation:
- **Notebook**: `01_data_preparation.ipynb`
- **Data Used**: 92 major world tourist cities, scraped live from Geoapify/OpenStreetMap and Wikidata.
- **Data Processing**:
  - Looked up each city's population by coordinates (not by name to avoid any false match or mismatch).
  - Scaled each city's search radius by population.
  - Pulled amenity density (dining, hotel, healthcare, transit, etc.), wheelchair-accessibility tag rates, and street-level mobility (% paved, % good smoothness).
### Model Training:
- **Notebook**: `02_modeling.ipynb`
- Built a single `mobility_score` (0-100) using PCA on 7 core accessibility signals (dining/hotel/attraction access, transit, toilets, street paving, street smoothness).
- Trained a Random Forest to predict that score using 4 *different* secondary features (parking, healthcare, essential shops, benches) — deliberately separate from the features that built the score, so the model is actually being tested on something meaningful and handing down the score.
- Clustered cities into 5 "mobility archetypes" based on those same 4 secondary features.

#### Results & Observation

| Model | Test MAE | Test R² (cross-validated) |
|---|---|---|
| Linear Regression | 14.17 | 0.388 |
| Ridge Regressor | 14.11 | 0.398 |
| **Random Forest (tuned)** | **13.97** | **0.443** |

- Random Forest performed best, using the cross-validated R² (0.44) over single split R² (0.57) we get after tuning. Because of the size of dataset cross validation is the more honest number.
- SHAP analysis showed `parking_density` and `bench_density` mattered far more to predictions than `healthcare_density` or `essential_density` — both parking and benches relate directly to physical mobility, which makes sense for this specific score.

### Deployment
- The Streamlit app takes any city name. If it's one of the 92 in the training data, it returns an instant result. If not, it fetches live data and predicts the score on the spot (takes 15-20 seconds).

## Key Decisions & Findings
- **PCA target only explains `~33%` of variance across its 7 input features** — meaning the score is a useful summary, but the underlying features measures fairly distinct things.
- **Population lookup by coordinates, not by name** — this fixes any false match or mismatches of location between Geoapify and Wikidata.
- **Tested 5 accessibility-related OSM tags before building features from any of them** — only `wheelchair` (34% coverage) and street `surface`/`smoothness` (92%/51% coverage) had real signal; `kerb`, `width`, `elevator`, `tactile_paving`, and `ramp` were all under 1% and excluded.
- **Group A / Group B split kept strictly separate** — there is no features leak, as the model never sees the features used for building target score and uses completely different group of feature to predict as its own inputs.

## Launching
```bash
pip install -r requirements.txt
streamlit run app.py
```
You'll need a Geoapify API key (free tier available). Put it in a `.env` file:
```
GEOAPIFY_KEY=your_key_here
```

## Limitation & Future Improvements
- There are still some issues with extracting data from Wikidata and Geoapify. Mainly due to mismatch in there name entry due to misspelling or completely different name. So 8 of the original planned cities were excluded.
- The live app's uses prediction model on secondary infrastructure (parking, healthcare, essential shops, benches) as input — it doesn't use the richer signals (dining/hotel/attraction accessibility, street paving/smoothness) that built the original mobility_score.
- More specific accessibility tags (curb cuts, street width) were tested and found to have almost no coverage in OpenStreetMap — a real data gap, not something this project could fix.
- Clustering used only the 4 secondary features, to keep it conceptually separate from the score-building features — clustering on the full feature set might reveal richer city archetypes, at the cost of a less independent second perspective.
- With only 92 cities and 4 predictor features, this is a small-sample model — results should be read as a reasonable estimate, not a precise measurement.