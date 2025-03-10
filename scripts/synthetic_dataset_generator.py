"""
SMART COMPOST - MODEL PROJECT.

---  Smart Compost Scripts 🌿
---   scripts/synthetic_data_generator.py

* Author  -  enos muthiani
* git     -  https://github.com/lyznne
* date    - 10 Mar 2025
* email   - emuthiani26@gmail.com


                                    Copyright (c) 2024      - enos.vercel.app
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random, os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(DATA_DIR, exist_ok=True)


def get_next_dataset_filename():
    """Finds the next dataset version (increments file number) in 'data/'."""
    i = 102
    while os.path.exists(os.path.join(DATA_DIR, f"smart_compost_dataset{i}.csv")):
        i += 1
    return os.path.join(DATA_DIR, f"smart_compost_dataset{i}.csv")


# Define composting variables and their relationships
VARIABLES = {
    "Temperature": {"unit": "Celsius", "optimal_range": "45-65"},
    "Moisture_Content": {"unit": "Percentage", "optimal_range": "40-60"},
    "pH_Level": {"unit": "pH Scale", "optimal_range": "6.5-8.0"},
    "Oxygen_Level": {"unit": "Percentage", "optimal_range": "5-15"},
    "Carbon_Nitrogen_Ratio": {"unit": "Ratio", "optimal_range": "25-30"},
    "Nitrogen_Content": {"unit": "Percentage", "optimal_range": "1.5-2.0"},
    "Potassium_Content": {"unit": "Percentage", "optimal_range": "0.5-1.5"},
    "Phosphorus_Content": {"unit": "Percentage", "optimal_range": "0.3-1.0"},
    "Ambient_Humidity": {"unit": "Percentage", "optimal_range": "40-70"},
    "Ambient_Temperature": {"unit": "Celsius", "optimal_range": "15-35"},
    "Waste_Height": {"unit": "Centimeters", "optimal_range": "90-150"},
    "Particle_Size": {"unit": "Millimeters", "optimal_range": "5-50"},
    "Bulk_Density": {"unit": "kg/m³", "optimal_range": "300-650"},
    "Final_Volume_Reduction": {"unit": "Percentage", "optimal_range": "40-60"},
    "Odor_Level": {"unit": "Scale 1-5", "optimal_range": "1-5"},
    "Decomposition_Rate": {"unit": "Percentage/Day", "optimal_range": "1-3"},
    "Time_Elapsed": {"unit": "Days", "optimal_range": "30-90"},
    "Maturity_Index": {"unit": "Scale 1-8", "optimal_range": "1-8"},
    "Turning_Frequency": {"unit": "Days", "optimal_range": "3-7"},
}

# Define seasonal effects
SEASONS = {
    "Spring": {"temp_change": 1.2, "moisture_change": 1.1},
    "Summer": {"temp_change": 1.5, "moisture_change": 0.8},
    "Fall": {"temp_change": 0.9, "moisture_change": 1.0},
    "Winter": {"temp_change": 0.7, "moisture_change": 1.3},
}

# Define waste types and their properties
WASTE_TYPES = {
    "GREEN": {"nitrogen": 2.0, "moisture": 60, "decomposition_rate": 2.5},
    "BROWN": {"nitrogen": 1.5, "moisture": 40, "decomposition_rate": 1.5},
    "MIXED": {"nitrogen": 1.8, "moisture": 50, "decomposition_rate": 2.0},
}


# Function to get the season
def get_season(date):
    month = date.month
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Fall"
    else:
        return "Winter"


# Generate synthetic dataset
def generate_synthetic_compost_data(days=365):
    start_date = datetime.now() - timedelta(days=days)
    data = []

    for i in range(days):
        date = start_date + timedelta(days=i)
        season = get_season(date)
        season_factors = SEASONS[season]
        waste_type = random.choice(list(WASTE_TYPES.keys()))
        waste_properties = WASTE_TYPES[waste_type]

        row = {"Date": date.strftime("%Y-%m-%d"), "Waste_Type": waste_type}

        # Generate base values for each variable
        for var, properties in VARIABLES.items():
            low, high = map(float, properties["optimal_range"].split("-"))
            value = np.random.uniform(low, high)

            # Adjust based on waste type
            if var == "Nitrogen_Content":
                value = waste_properties["nitrogen"]
            elif var == "Moisture_Content":
                value = waste_properties["moisture"]
            elif var == "Decomposition_Rate":
                value = waste_properties["decomposition_rate"]

            # Introduce seasonal variations
            if "Temperature" in var:
                value *= season_factors["temp_change"]
            if "Moisture" in var or "Humidity" in var:
                value *= season_factors["moisture_change"]

            # Add noise
            noise = np.random.normal(0, (high - low) * 0.05)
            row[var] = round(max(low, min(high, value + noise)), 2)

        # Add derived variables
        row["Final_Volume_Reduction"] = (
            row["Decomposition_Rate"] * row["Time_Elapsed"] / 100
        )
        row["Maturity_Index"] = min(
            8, row["Time_Elapsed"] / 10 + row["Decomposition_Rate"]
        )

        data.append(row)

    return pd.DataFrame(data)


# Generate dataset
df = generate_synthetic_compost_data()

# Get the next available filename
dataset_path = get_next_dataset_filename()

# Save dataset in 'data/' folder
df.to_csv(dataset_path, index=False)
print(f"Synthetic compost dataset saved at:\n📂 {dataset_path}")
