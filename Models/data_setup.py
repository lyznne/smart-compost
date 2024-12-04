"""
SMART COMPOST - MODEL PROJECT.

---  where dataset is prepared
---   Models/data_setup.py

* Author  -  enos muthiani
* git     -  https://github.com/lyznne
* date    - 3 Dec 2024
* email   - emuthiani26@gmail.com


                                    Copyright (c) 2024      - enos.vercel.app
"""

# imports
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import  Dataset


class WasteType:
    """Classification of different waste types and their properties"""

    TYPES = {
        "GREEN": {
            "examples": ["grass", "vegetable_scraps", "coffee_grounds", "fruit_waste"],
            "nitrogen_content": "high",
            "moisture_content": "high",
            "decomposition_rate": "fast",
        },
        "BROWN": {
            "examples": ["leaves", "straw", "paper", "cardboard", "wood_chips"],
            "nitrogen_content": "low",
            "moisture_content": "low",
            "decomposition_rate": "slow",
        },
        "MIXED": {
            "examples": ["food_scraps", "garden_waste"],
            "nitrogen_content": "medium",
            "moisture_content": "medium",
            "decomposition_rate": "medium",
        },
    }

    @staticmethod
    def get_type_encoding(waste_type: str) -> torch.Tensor:
        """Convert waste type to one-hot encoding"""
        encoding = torch.zeros(len(WasteType.TYPES))

        if waste_type in WasteType.TYPES:
            encoding[list(WasteType.TYPES.keys()).index(waste_type)] = 1

        return encoding


class SeasonalEffects:
    """Handle seasonal and weather effects on composting"""

    SEASONS = {
        "SPRING": {"temp_modifier": 0.2, "moisture_modifier": 0.3},
        "SUMMER": {"temp_modifier": 0.4, "moisture_modifier": -0.2},
        "FALL": {"temp_modifier": 0.0, "moisture_modifier": 0.1},
        "WINTER": {"temp_modifier": -0.3, "moisture_modifier": 0.2},
    }

    @staticmethod
    def get_season(date: datetime) -> str:
        """Determine season from date"""
        month = date.month
        if month in [3, 4, 5]:
            return "SPRING"
        elif month in [6, 7, 8]:
            return "SUMMER"
        elif month in [9, 10, 11]:
            return "FALL"
        else:
            return "WINTER"


class CompostTimeSeriesDataset(Dataset):
    """Compost Time Series for our Dataset

    Args:
        Dataset uses PyTorch Dataset
    """

    def __init__(self, csv_path: str, sequence_length: int = 30):
        super().__init__()
        self.sequence_length = sequence_length
        self.base_data = pd.read_csv(csv_path, delimeter=",", skip_blank_lines=True)
        self.base_data.columns = self.base_data.columns.str.strip()

        # Initialize time series data structures
        self.time_series_data = self._generate_time_series_data()
        self.waste_types = self._initialize_waste_types()
        self.weather_data = self._initialize_weather_data()

        # Process optimal ranges
        self._process_optimal_ranges()

        # Create feature matrices
        self._create_feature_matrices()

    def _generate_time_series_data(self) -> Dict[str, List[float]]:
        """Generate synthetic time series data for each variable

        Returns:
            Dict[str, List[float]]: time series data
        """
        time_series = {}
        start_date = datetime.now() - timedelta(days=365)
        dates = [start_date + timedelta(days=i) for i in range(365)]

        for _, row in self.base_data.iterrows():
            variable = row["Variable"]
            optimal_range = self._parse_range(row["OptimalRange"])

            if optimal_range:
                mid_point = (optimal_range[0] + optimal_range[1]) / 2
                # Generate daily values with realistic Variations
                values = [
                    self._generate_realistic_value(mid_point, optimal_range, dates[i])
                    for i in range(365)
                ]
                time_series[variable] = values

        return time_series

    def _generate_realistic_value(
        self, mid_point: float, optimal_range: Tuple[float, float], date: datetime
    ) -> float:
        """Generate realistic value considering seasonal effects and daily variations

        Args:
            mid_point (float): optimal / midpoint value
            optimal_range (Tuple[float, float]): range where the variable works best
            date (datetime): time

        Returns:
            float: Optimal range value
        """
        season = SeasonalEffects.get_season(date)
        season_effect = SeasonalEffects.SEASONS[season]["temp_modifier"]

        # add daily variation ( random walk with bounds )
        daily_variation = np.random.normal(0, 0.05)
        value = mid_point * (1 + season_effect + daily_variation)

        # Ensure value stays within optimal range
        return max(optimal_range[0], min(optimal_range[1], value))

    def _initialize_waste_types(self) -> List[str]:
        """Initialize waste type sequence

        Returns:
            List[str]: waste type
        """

        return np.random.choice(list(WasteType.TYPES.keys()), size=365).tolist()

    def _initialize_weather_data(self) -> Dict[str, List[float]]:
        """Generate synthetic weather data

        Returns:
            Dict[str, List[float]]: type of weather converts to weather data
        """

        weather_data = {"temperature": [], "humidity": [], "rainfall": []}

        start_date = datetime.now() - timedelta(days=365)
        for i in range(365):
            date = start_date + timedelta(days=i)
            season = SeasonalEffects.get_season(date)
            season_effects = SeasonalEffects.SEASONS[season]

            # Generate weather data with seasonal effects
            weather_data["temperature"].append(
                20 + season_effects["temp_modifier"] * 30 + np.random.normal(0, 2)
            )
            weather_data["humidity"].append(
                60 + season_effects["moisture_modifier"] * 30 + np.random.normal(0, 5)
            )
            weather_data["rainfall"].append(
                max(
                    0,
                    np.random.exponential(scale=2)
                    * (1 + season_effects["moisture_modifier"]),
                )
            )

        return weather_data

    def _parse_range(self, range_str: str) -> Tuple[float, float]:
        """Parse range string to tuple of floats

        Args:
            range_str (str): variable with range

        Returns:
            Tuple[float, float]: range values
        """
        try:
            if isinstance(range_str, str):
                range_str = range_str.strip()
                if ">" in range_str:
                    val = float(range_str.replace(">", ""))
                    return (val, val * 1.2)
                elif ":" in range_str:
                    range_part = range_str.split(":")[0]
                    low, high = map(float, range_part.split("-"))
                    return (low, high)
                else:
                    return tuple(map(float, range_str.split("-")))
            return None
        except:
            return None

    def _process_optimal_ranges(self):
        """Process optimal ranges for all variables"""
        self.optimal_ranges = {}
        for _, row in self.base_data.iterrows():
            range_vals = self._parse_range(row["OptimalRange"])
            if range_vals:
                self.optimal_ranges[row["Variable"]] = range_vals

    def _create_feature_matrices(self):
        """Create feature matrices for the dataset"""
        self.features = []
        self.targets = []

        for i in range(
            len(self.time_series_data["Temperature"]) - self.sequence_length
        ):
            # Create sequence of features
            sequence = []
            for day in range(self.sequence_length):
                idx = i + day

                # Measurements
                measurements = [
                    self.time_series_data[var][idx]
                    for var in self.time_series_data.keys()
                ]

                # Waste type encoding
                waste_encoding = WasteType.get_type_encoding(self.waste_types[idx])

                # Weather data
                weather = [self.weather_data[w][idx] for w in self.weather_data.keys()]

                # Combine all features
                day_features = measurements + waste_encoding.tolist() + weather
                sequence.append(day_features)

            self.features.append(torch.tensor(sequence, dtype=torch.float32))

            # Target is the next day's temperature and moisture content
            target_idx = i + self.sequence_length
            target = [
                self.time_series_data["Temperature"][target_idx],
                self.time_series_data["Moisture_Content"][target_idx],
            ]
            self.targets.append(torch.tensor(target, dtype=torch.float32))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

    def get_optimal_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Return optimal ranges for all variables"""
        return self.optimal_ranges

    def get_variable_metadata(self) -> pd.DataFrame:
        """Return metadata for all variables"""
        return self.base_data[
            [
                "Variable",
                "Type",
                "Unit",
                "Dependencies",
                "IntroductionStage",
                "Frequency",
                "Notes",
            ]
        ]
