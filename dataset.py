"""Generates the 10,000-row crime dataset used by the app.

Run `python dataset.py` to also write crime_data_10000.csv next to this file.
"""
import numpy as np
import pandas as pd

CITIES = ["Pune", "Mumbai", "Nashik", "Nagpur", "Delhi",
          "Bangalore", "Hyderabad", "Chennai", "Ahmedabad", "Jaipur"]
CITY_EFFECT = {"Pune": -0.15, "Mumbai": 0.10, "Nashik": -0.25, "Nagpur": -0.05,
               "Delhi": 0.35, "Bangalore": 0.05, "Hyderabad": 0.00,
               "Chennai": -0.10, "Ahmedabad": -0.20, "Jaipur": 0.10}

AREAS = ["Residential", "Commercial", "Industrial",
         "Informal Settlement", "Transit Hub", "Mixed Use"]
AREA_P = [.30, .18, .10, .14, .08, .20]
AREA_EFFECT = {"Residential": -0.30, "Commercial": 0.20, "Industrial": 0.00,
               "Informal Settlement": 0.60, "Transit Hub": 0.50, "Mixed Use": 0.10}
AREA_DENSITY = {"Residential": 9000, "Commercial": 12000, "Industrial": 4000,
                "Informal Settlement": 26000, "Transit Hub": 15000, "Mixed Use": 11000}
AREA_UNEMP = {"Residential": 7, "Commercial": 6, "Industrial": 9,
              "Informal Settlement": 14, "Transit Hub": 10, "Mixed Use": 8}
AREA_LIT = {"Residential": 90, "Commercial": 91, "Industrial": 82,
            "Informal Settlement": 71, "Transit Hub": 80, "Mixed Use": 86}
AREA_CCTV = {"Residential": 35, "Commercial": 62, "Industrial": 28,
             "Informal Settlement": 12, "Transit Hub": 55, "Mixed Use": 40}
AREA_INCOME = {"Residential": 38, "Commercial": 55, "Industrial": 30,
               "Informal Settlement": 16, "Transit Hub": 28, "Mixed Use": 36}

CRIMES = ["Theft", "Burglary", "Assault", "Robbery",
          "Vehicle Theft", "Cyber Fraud", "Vandalism", "Drug Offence"]
CRIME_BY_AREA = {
    "Residential":         [.18, .20, .10, .05, .14, .14, .12, .07],
    "Commercial":          [.20, .08, .08, .14, .10, .26, .09, .05],
    "Industrial":          [.12, .14, .16, .08, .16, .05, .15, .14],
    "Informal Settlement": [.14, .12, .24, .10, .06, .04, .12, .18],
    "Transit Hub":         [.30, .04, .12, .16, .14, .06, .08, .10],
    "Mixed Use":           [.18, .12, .12, .10, .12, .14, .12, .10],
}
TIMES = ["Morning", "Afternoon", "Evening", "Night"]
TIME_BY_CRIME = {
    "Theft":         [.15, .30, .35, .20],
    "Burglary":      [.10, .18, .12, .60],
    "Assault":       [.05, .12, .33, .50],
    "Robbery":       [.05, .10, .30, .55],
    "Vehicle Theft": [.12, .14, .24, .50],
    "Cyber Fraud":   [.30, .35, .25, .10],
    "Vandalism":     [.08, .14, .28, .50],
    "Drug Offence":  [.06, .12, .32, .50],
}
DAY_TYPES = ["Weekday", "Weekend", "Holiday"]


def _z(x):
    return (x - x.mean()) / x.std()


def _sample_by_group(rng, keys, table, labels):
    """Draw one label per row, using the probability row that belongs to its key."""
    out = np.empty(len(keys), dtype=object)
    for k, probs in table.items():
        idx = np.where(keys == k)[0]
        p = np.array(probs, dtype=float)
        out[idx] = rng.choice(labels, len(idx), p=p / p.sum())
    return out


def create_dataset(n=10000, seed=42):
    rng = np.random.default_rng(seed)

    city = rng.choice(CITIES, n)
    area = rng.choice(AREAS, n, p=AREA_P)
    year = rng.integers(2019, 2026, n)
    month = rng.integers(1, 13, n)
    day_type = rng.choice(DAY_TYPES, n, p=[.68, .24, .08])
    crime = _sample_by_group(rng, area, CRIME_BY_AREA, CRIMES)
    time_of_day = _sample_by_group(rng, crime, TIME_BY_CRIME, TIMES)

    def per_area(table):
        return np.array([table[a] for a in area], dtype=float)

    density = np.clip(per_area(AREA_DENSITY) * rng.lognormal(0, .35, n), 500, 45000)
    unemp = np.clip(per_area(AREA_UNEMP) + rng.normal(0, 2.2, n), 2, 25)
    literacy = np.clip(per_area(AREA_LIT) + rng.normal(0, 4.5, n), 45, 99)
    police = np.clip(rng.normal(130, 40, n), 30, 300)
    cctv = np.clip(per_area(AREA_CCTV) + rng.normal(0, 14, n), 0, 95)
    lighting = np.clip(rng.normal(75, 14, n), 25, 100)
    income = np.clip(per_area(AREA_INCOME) * rng.lognormal(0, .30, n), 8, 150)

    # Hidden pressure score that drives both the crime rate and the risk label
    z = (.55 * _z(np.log(density)) + .45 * _z(unemp) - .35 * _z(police)
         - .40 * _z(literacy) - .30 * _z(cctv) - .25 * _z(lighting)
         - .35 * _z(np.log(income)))
    z = z + np.array([AREA_EFFECT[a] for a in area]) + np.array([CITY_EFFECT[c] for c in city])
    z = z + rng.normal(0, .25, n)

    crime_rate = np.clip(np.exp(4.2 + .55 * z + rng.normal(0, .25, n)), 5, 500)
    prior = np.clip(rng.poisson(np.exp(1.6 + .35 * z)), 0, 60)

    p_high = 1 / (1 + np.exp(-1.4 * (z - np.median(z) - .1)))
    risk = np.where(rng.random(n) < p_high, "High Risk", "Low Risk")

    df = pd.DataFrame({
        "Record_ID": [f"R{i:05d}" for i in range(1, n + 1)],
        "City": city, "Area_Type": area, "Year": year, "Month": month,
        "Day_Type": day_type, "Crime_Type": crime, "Time_of_Day": time_of_day,
        "Population_Density": np.round(density, 0),
        "Unemployment_Rate": np.round(unemp, 1),
        "Literacy_Rate": np.round(literacy, 1),
        "Police_Per_100k": np.round(police, 0),
        "CCTV_Coverage": np.round(cctv, 1),
        "Street_Lighting": np.round(lighting, 1),
        "Avg_Income_K": np.round(income, 1),
        "Prior_Incidents": prior,
        "Crime_Rate": np.round(crime_rate, 2),
        "Risk_Level": risk,
    })
    # a few gaps so the preprocessing module has real work to do
    for col in ["Population_Density", "Literacy_Rate", "CCTV_Coverage"]:
        idx = rng.choice(n, 20, replace=False)
        df.loc[idx, col] = np.nan
    return df


if __name__ == "__main__":
    data = create_dataset(10000)
    data.to_csv("crime_data_10000.csv", index=False)
    print(data.shape)
    print(data.Risk_Level.value_counts())
