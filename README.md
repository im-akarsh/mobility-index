# City Mobility & Accessibility Score
Scoring how walkable and accessible a city is for a disabled traveler, using amenity density, accessibility tags, and street-level walkability data pulled live from OpenStreetMap (via Geoapify). Built a model that predicts this score from secondary infrastructure alone, and clustered cities into mobility "archetypes."

## Links
[Live Web App]

## Objective
When someone with a disability is planning to travel, it's genuinely hard to know how accessible a city actually is before you get there. This project scores cities on accessibility and walkability using real, publicly available map data, and checks whether a smaller set of secondary infrastructure signals (parking, healthcare, shops, benches) can predict that score on their own.

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
- **Data Used**: 92 major world tourist cities (originally targeting 100, 8 excluded — see Limitations), scraped live from Geoapify/OpenStreetMap and Wikidata.
- **Data Processing**:
  - Looked up each city's population by coordinates (not by name), since Geoapify and Wikidata often use different names for the same place (e.g. "Bangalore" vs. "Bengaluru").
  - Scaled each city's search radius by population.
  - Pulled amenity density (dining, hotel, healthcare, transit, etc.), wheelchair-accessibility tag rates, and street-level walkability (% paved, % good smoothness).

### Debugging Real Data Issues
This part took a while, and honestly taught me more than the modeling did.

While testing city name lookups, I found real false-match bugs — searching "Madrid" was matching a tiny village also named Madrid (population: 2,802, not the real ~3.3 million), same for Krakow and Mugla. I fixed this by checking multiple name variants (accented and not) and keeping whichever match had the largest population, with a minimum threshold to filter out obviously-wrong tiny matches.

A separate problem: some cities (like Bangalore) kept failing entirely, because Geoapify and Wikidata just use different names for the same place. I fixed this by looking up population by the city's actual coordinates instead of its name — sidesteps the naming mismatch completely.

I also tested which OpenStreetMap accessibility tags were actually usable before building features from them. `wheelchair` had real coverage (~34% of places). `elevator`, `tactile_paving`, `ramp`, `kerb`, and `width` all tested at under 1% coverage — basically unrecorded in the data — so I didn't build features from those at all, rather than pretend they carried signal they don't have.

### Model Training:
- **Notebook**: `02_modeling.ipynb`
- Built a single `mobility_score` (0-100) using PCA on 7 core accessibility signals (dining/hotel/attraction access, transit, toilets, street paving, street smoothness).
- Trained a Random Forest to predict that score using 4 *different* secondary features (parking, healthcare, essential shops, benches) — deliberately separate from the features that built the score, so the model is actually being tested on something meaningful, not just handed its own answer key.
- Clustered cities into 5 "mobility archetypes" based on those same 4 secondary features.

#### Results & Observation

| Model | Test MAE | Test R² (cross-validated) |
|---|---|---|
| Linear Regression | 14.17 | 0.388 |
| Ridge Regressor | 14.11 | 0.398 |
| **Random Forest (tuned)** | **13.97** | **0.443** |

- Random Forest performed best, though I'm reporting the cross-validated R² (0.44) rather than the single test-split R² (0.57), since with only ~18 cities in the test split, one lucky split can be misleading — cross-validation is the more honest number.
- SHAP analysis showed `parking_density` and `bench_density` mattered far more to predictions than `healthcare_density` or `essential_density` — both parking and benches relate directly to physical mobility, which makes sense for this specific score.

### Deployment
- The Streamlit app takes any city name. If it's one of the 92 in the training data, it returns an instant result. If not, it fetches live data and predicts the score on the spot (takes 15-20 seconds).

## Key Decisions & Findings
- **PCA target only explains ~33% of variance across its 7 input features** — meaning the mobility_score is a useful summary, but the underlying features (walkability, dining access, transit, etc.) measure fairly distinct things, not one tightly correlated dimension.
- **Population lookup by coordinates, not by name** — fixes real naming mismatches between Geoapify and Wikidata (Bangalore/Bengaluru, Ha Long matching the wrong commune, etc.) at the source, rather than patching each mismatch individually.
- **Tested 5 accessibility-related OSM tags before building features from any of them** — only `wheelchair` (34% coverage) and street `surface`/`smoothness` (92%/51% coverage) had real signal; `kerb`, `width`, `elevator`, `tactile_paving`, and `ramp` were all under 1% and excluded.
- **Group A / Group B split kept strictly separate** — the model never sees the features that built the target, so its ~44% R² reflects genuine predictive signal from secondary infrastructure, not the model just re-deriving its own inputs.

## Launching
```bash
pip install -r requirements.txt
streamlit run app.py
```
You'll need a Geoapify API key (free tier available). Put it in a `.env` file:
```
GEOAPIFY_API_KEY=your_key_here
```

## Limitation & Future Improvements
- 8 of the originally planned 100 cities (including Hong Kong and Macau) couldn't be resolved — likely due to how Wikidata classifies Special Administrative Regions differently from typical cities. Excluded rather than force-fit.
- The live app's prediction model only uses secondary infrastructure (parking, healthcare, essential shops, benches) as input — it doesn't use the richer signals (dining/hotel/attraction accessibility, street paving/smoothness) that built the original mobility_score.
- More specific accessibility tags (curb cuts, street width) were tested and found to have almost no coverage in OpenStreetMap — a real data gap, not something this project could fix.
- Clustering used only the 4 secondary features, to keep it conceptually separate from the score-building features — clustering on the full feature set might reveal richer city archetypes, at the cost of a less independent second perspective.
- With only 92 cities and 4 predictor features, this is a small-sample model — results should be read as a reasonable estimate, not a precise measurement.