import os
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import joblib
import shap
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEOAPIFY_KEY")

GEOCODE_URL = "https://api.geoapify.com/v1/geocode/search"
PLACES_URL = "https://api.geoapify.com/v2/places"
WIKIDATA_URL = "https://query.wikidata.org/sparql"
WIKIDATA_HEADERS = {"User-Agent": "CityWalkMobilityIQ/1.0"}

GROUP_B_COLS = ["parking_density", "healthcare_density", "essential_density", "bench_density"]

# Load trained model files
rf_model = joblib.load("models/mobility_rf_model.pkl")
scaler = joblib.load("models/mobility_scaler.pkl")
kmeans = joblib.load("models/mobility_kmeans.pkl")
explainer = shap.TreeExplainer(rf_model)

# Load the cities we already have scores for
scored_cities = pd.read_csv("data/processed/city_accessibility_scored.csv")

st.title("City Mobility Score")
st.write("Type a city name to see its predicted mobility/accessibility score.")

city_input = st.text_input("City name", placeholder="e.g. Paris, France")


def safe_get(url, params, headers=None):
    for attempt in range(3):
        try:
            return requests.get(url, params=params, headers=headers, timeout=45)
        except requests.exceptions.RequestException:
            time.sleep(3)
    return None


def get_population(lat, lon):
    query = f"""
    SELECT ?population WHERE {{
        SERVICE wikibase:around {{
            ?city wdt:P625 ?location.
            bd:serviceParam wikibase:center "Point({lon} {lat})"^^geo:wktLiteral.
            bd:serviceParam wikibase:radius "10".
        }}
        ?city wdt:P31/wdt:P279* wd:Q515.
        ?city wdt:P1082 ?population.
    }}
    ORDER BY DESC(?population)
    LIMIT 1
    """
    response = safe_get(WIKIDATA_URL, params={"query": query, "format": "json"}, headers=WIKIDATA_HEADERS)
    if response is None:
        return None
    results = response.json().get("results", {}).get("bindings", [])
    if results:
        return int(results[0]["population"]["value"])
    return None


def get_places_count(lon, lat, radius_m, category):
    params = {
        "categories": category,
        "filter": f"circle:{lon},{lat},{radius_m}",
        "limit": 500,
        "apiKey": API_KEY,
    }
    response = safe_get(PLACES_URL, params)
    if response is None:
        return 0
    return len(response.json().get("features", []))


def get_places_features(lon, lat, radius_m, category):
    params = {
        "categories": category,
        "filter": f"circle:{lon},{lat},{radius_m}",
        "limit": 500,
        "details": "details.accessibility",
        "apiKey": API_KEY,
    }
    response = safe_get(PLACES_URL, params)
    if response is None:
        return []
    return response.json().get("features", [])


def get_accessibility_tag_coverage(places):
    """% of places that have a wheelchair tag at all, used as a data-confidence signal."""
    if not places:
        return 0.0
    tagged = sum(
        1 for p in places
        if p.get("properties", {}).get("datasource", {}).get("raw", {}).get("wheelchair") is not None
    )
    return round((tagged / len(places)) * 100, 2)


def get_city_data_live(city_name):
    # Step 1: find the city's coordinates
    geo_params = {"text": city_name, "type": "city", "apiKey": API_KEY}
    geo_response = safe_get(GEOCODE_URL, params=geo_params)
    if not geo_response or not geo_response.json().get("features"):
        return None

    props = geo_response.json()["features"][0]["properties"]
    lon, lat = props["lon"], props["lat"]

    # Step 2: find its population, to set a search radius
    population = get_population(lat, lon)
    if population is None:
        return None
    radius_m = 8000  # keep it simple, use a fixed 8km radius for live lookups

    # Step 3: count the 4 features our model needs
    area_km2 = np.pi * (radius_m / 1000) ** 2
    categories = {
        "parking_density": "parking",
        "healthcare_density": "healthcare",
        "essential_density": "commercial.supermarket,commercial.convenience",
        "bench_density": "leisure.park",
    }

    city_data = {"city": city_name, "population": population}
    for feature_name, category_string in categories.items():
        count = get_places_count(lon, lat, radius_m, category_string)
        city_data[feature_name] = round(count / area_km2, 2)

    # Step 4: check accessibility tag coverage for restaurants, as a
    # rough data-confidence signal (we know from testing that this tag
    # is only present on ~34% of places on average — cities far below
    # that should be flagged as lower-confidence)
    dining_places = get_places_features(lon, lat, radius_m, "catering.restaurant")
    city_data["tag_coverage_pct"] = get_accessibility_tag_coverage(dining_places)

    return city_data


if st.button("Get Score"):
    if not city_input:
        st.warning("Please type a city name first.")
    else:
        # Check if we already have this city in our scored dataset
        match = scored_cities[scored_cities["city"].str.lower() == city_input.lower()]

        if not match.empty:
            st.success("Found this city in our dataset — instant result!")
            row = match.iloc[0]
            score = row["mobility_score"]
            cluster = int(row["mobility_cluster"])
            population = row["population"]
            tag_coverage = row.get("tag_coverage_pct", 0)

            X_scaled = scaler.transform(pd.DataFrame([row[GROUP_B_COLS].values], columns=GROUP_B_COLS))
            shap_values = explainer.shap_values(X_scaled)[0]

        else:
            st.info("City not in our dataset. Fetching live data, this may take 15-20 seconds...")
            city_data = get_city_data_live(city_input)

            if city_data is None:
                st.error("Could not find this city. Try adding the country name.")
                st.stop()

            X = pd.DataFrame([[city_data[col] for col in GROUP_B_COLS]], columns=GROUP_B_COLS)
            X_scaled = scaler.transform(X)

            score = round(float(rf_model.predict(X_scaled)[0]), 1)
            cluster = int(kmeans.predict(X_scaled)[0])
            population = city_data["population"]
            tag_coverage = city_data["tag_coverage_pct"]
            shap_values = explainer.shap_values(X_scaled)[0]

        # --- Results ---
        st.subheader(city_input.title())

        col1, col2, col3 = st.columns(3)
        col1.metric("Mobility Score", f"{score} / 100")
        col2.metric("Cluster Group", cluster)
        confidence_label = "Low" if tag_coverage < 15 else "OK"
        col3.metric("Data Confidence", f"{tag_coverage:.1f}%", delta=confidence_label, delta_color="off")

        st.write(f"Population: {population:,}")

        # --- Rank among training data ---
        rank_position = (scored_cities["mobility_score"] > score).sum() + 1
        total_cities = len(scored_cities)
        st.write(f"Rank: **#{rank_position} of {total_cities}** cities in our training data, by mobility score.")

        # --- What's driving this score (SHAP) ---
        st.subheader("What's driving this score")
        top_factors = sorted(zip(GROUP_B_COLS, shap_values), key=lambda x: -abs(x[1]))[:3]
        for feature, impact in top_factors:
            direction = "increased" if impact > 0 else "decreased"
            st.write(f"- **{feature.replace('_', ' ').title()}** {direction} the score ({impact:+.2f})")

        # --- Comparison chart ---
        st.subheader("How this city compares")
        comparison_df = scored_cities[["city", "mobility_score"]].copy()
        if match.empty:
            comparison_df = pd.concat(
                [comparison_df, pd.DataFrame([{"city": city_input, "mobility_score": score}])],
                ignore_index=True,
            )
        comparison_df = comparison_df.sort_values("mobility_score")
        st.bar_chart(comparison_df, x="city", y="mobility_score")

        # --- Limitations ---
        with st.expander("Read before trusting this score (limitations)"):
            st.markdown("""
            - This model predicts mobility score from secondary infrastructure only
              (parking, healthcare, essential shops, benches) — it doesn't directly
              measure things like ramps, curb cuts, or street width.
            - Accessibility tag data comes from OpenStreetMap, which is crowdsourced
              and incomplete — cities with a low Data Confidence score have less
              reliable underlying data.
            - Cities not in the original dataset are scored live, using the same
              model — treat these as estimates.
            """)