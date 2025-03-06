import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random, os


# Set base directory to save datasets in 'data/' (one level up from scripts/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(DATA_DIR, exist_ok=True)

def get_next_dataset_filename():
    """Finds the next dataset version (increments file number) in 'data/'."""
    i = 102
    while os.path.exists(os.path.join(DATA_DIR, f"smart_compost_dataset{i}.csv")):
        i += 1
    return os.path.join(DATA_DIR, f"smart_compost_dataset{i}.csv")

# Define composting variables
VARIABLES = {
    "Temperature": (45, 65),  # Optimal range in Celsius
    "Moisture_Content": (40, 60),  # Percentage
    "pH_Level": (6.5, 8.0),  # pH Scale
    "Oxygen_Level": (5, 15),  # Percentage
    "Carbon_Nitrogen_Ratio": (25, 30),  # Ratio
    "Nitrogen_Content": (1.5, 2.0),  # Percentage
    "Potassium_Content": (0.5, 1.5),  # Percentage
    "Phosphorus_Content": (0.3, 1.0),  # Percentage
    "Ambient_Humidity": (40, 70),  # Percentage
    "Ambient_Temperature": (15, 35),  # Celsius
    "Waste_Height": (90, 150),  # Centimeters
    "Particle_Size": (5, 50),  # Millimeters
    "Bulk_Density": (300, 650),  # kg/m³
    "Odor_Level": (1, 5),  # Scale 1-5
    "Decomposition_Rate": (1, 3),  # Percentage/Day
    "Time_Elapsed": (30, 90),  # Days
    "Maturity_Index": (7, 8),  # Scale 1-8
    "Turning_Frequency": (3, 7),  # Days
}

SEASONS = {
    "Spring": {"temp_change": 1.2, "moisture_change": 1.1},
    "Summer": {"temp_change": 1.5, "moisture_change": 0.8},
    "Fall": {"temp_change": 0.9, "moisture_change": 1.0},
    "Winter": {"temp_change": 0.7, "moisture_change": 1.3},
}

WASTE_TYPES = ["GREEN", "BROWN", "MIXED"]


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
        waste_type = random.choice(WASTE_TYPES)

        row = {"Date": date.strftime("%Y-%m-%d"), "Waste_Type": waste_type}

        for var, (low, high) in VARIABLES.items():
            # Generate realistic values
            value = np.random.uniform(low, high)

            # Introduce seasonal variations
            if "Temperature" in var:
                value *= season_factors["temp_change"]
            if "Moisture" in var or "Humidity" in var:
                value *= season_factors["moisture_change"]

            # Add noise
            noise = np.random.normal(0, (high - low) * 0.05)
            row[var] = round(max(low, min(high, value + noise)), 2)

        data.append(row)

    return pd.DataFrame(data)

# Generate dataset
df = generate_synthetic_compost_data()

# Get the next available filename
dataset_path = get_next_dataset_filename()

# Save dataset in 'data/' folder
df.to_csv(dataset_path, index=False)
print(f"✅ Synthetic compost dataset saved at:\n📂 {dataset_path}")
