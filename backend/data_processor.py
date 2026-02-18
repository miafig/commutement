"""Data processing and feature engineering for commute data."""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Any, List
from config import NUM_FEATURES, TOTAL_FEATURES


# Mapping for day names to indices
DAY_MAPPING = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6,
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

# Route name to ID mapping
ROUTE_MAPPING = {
    "paddington": 0,
    "monument": 1,
    "monument+walk": 1,
    "other": 2
}

# Stop name to ID mapping
STOPS_MAPPING = {
    "none": [],
    "coffee": [0],
    "lunch": [1],
    "coffee+lunch": [0, 1]
}


def time_str_to_minutes(time_str: str) -> int:
    """
    Convert time string (HH:MM) to minutes since midnight.

    Args:
        time_str: Time in format "HH:MM"

    Returns:
        Minutes since midnight (0-1440)
    """
    try:
        hours, minutes = map(int, time_str.split(":"))
        total_minutes = hours * 60 + minutes
        if total_minutes < 0 or total_minutes > 1440:
            raise ValueError(f"Invalid time: {time_str}")
        return total_minutes
    except ValueError:
        raise ValueError(f"Invalid time format: {time_str}. Expected HH:MM")


def normalize_time(time_value: int, min_val: int = 0, max_val: int = 1440) -> float:
    """
    Normalize time value (0-1440 minutes) to [0, 1].

    Args:
        time_value: Time in minutes (0-1440)
        min_val: Minimum time value
        max_val: Maximum time value

    Returns:
        Normalized time value
    """
    if time_value < min_val or time_value > max_val:
        raise ValueError(f"Time must be between {min_val} and {max_val} minutes")
    return (time_value - min_val) / (max_val - min_val)


def encode_day_of_week(day: str) -> np.ndarray:
    """
    One-hot encode day of week.

    Args:
        day: Day name (e.g., 'Monday')

    Returns:
        One-hot encoded vector of length 7
    """
    if day not in DAY_MAPPING:
        raise ValueError(f"Unknown day: {day}. Must be Monday-Sunday")

    idx = DAY_MAPPING[day]
    encoding = np.zeros(NUM_FEATURES["day_of_week"])
    encoding[idx] = 1
    return encoding


def encode_route(route_name: str) -> np.ndarray:
    """
    One-hot encode route.

    Args:
        route_name: Route name (e.g., 'paddington')

    Returns:
        One-hot encoded vector of length 3
    """
    route_name = route_name.lower().strip()
    if route_name not in ROUTE_MAPPING:
        raise ValueError(f"Unknown route: {route_name}. Must be one of {list(ROUTE_MAPPING.keys())}")

    route_id = ROUTE_MAPPING[route_name]
    encoding = np.zeros(NUM_FEATURES["route"])
    encoding[route_id] = 1
    return encoding


def encode_stops(stop_data: Any) -> np.ndarray:
    """
    Multi-hot encode stops.

    Args:
        stop_data: Stop info (can be list like ["coffee"] or encoded already)

    Returns:
        Multi-hot encoded vector of length 2
    """
    encoding = np.zeros(NUM_FEATURES["stops"])

    # Handle different input formats
    if isinstance(stop_data, list):
        # List of stop names
        for stop_name in stop_data:
            stop_name_lower = stop_name.lower().strip() if isinstance(stop_name, str) else "none"
            if stop_name_lower == "none":
                continue
            if stop_name_lower == "coffee":
                encoding[0] = 1
            elif stop_name_lower == "lunch":
                encoding[1] = 1
    elif isinstance(stop_data, str):
        stop_name_lower = stop_data.lower().strip()
        if stop_name_lower == "coffee":
            encoding[0] = 1
        elif stop_name_lower == "lunch":
            encoding[1] = 1

    return encoding


def commute_to_features(commute_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert raw commute data to feature vector and mask.

    Supports the actual frontend data format:
    {
        "dayOfWeek": "Monday",
        "departureTime": "08:00",
        "arrivalTime": "09:00",
        "route": "paddington",
        "stops": ["coffee"],
        "pleasureRating": 5,
        ...
    }

    Returns:
        Tuple of (features_vector, mask_vector) both of shape (TOTAL_FEATURES,)
    """
    features = []
    mask = []

    # Day of week
    if "dayOfWeek" in commute_data or "day" in commute_data:
        day_key = "dayOfWeek" if "dayOfWeek" in commute_data else "day"
        day_encoding = encode_day_of_week(commute_data[day_key])
        features.extend(day_encoding)
        mask.extend(np.ones(NUM_FEATURES["day_of_week"]))
    else:
        features.extend(np.zeros(NUM_FEATURES["day_of_week"]))
        mask.extend(np.zeros(NUM_FEATURES["day_of_week"]))

    # Start time (departureTime)
    if "departureTime" in commute_data or "startTime" in commute_data:
        time_key = "departureTime" if "departureTime" in commute_data else "startTime"
        time_val = commute_data[time_key]

        if isinstance(time_val, str):
            time_minutes = time_str_to_minutes(time_val)
        else:
            time_minutes = int(time_val)

        start_time_norm = normalize_time(time_minutes)
        features.append(start_time_norm)
        mask.append(1)
    else:
        features.append(0)
        mask.append(0)

    # End time (arrivalTime)
    if "arrivalTime" in commute_data or "endTime" in commute_data:
        time_key = "arrivalTime" if "arrivalTime" in commute_data else "endTime"
        time_val = commute_data[time_key]

        if isinstance(time_val, str):
            time_minutes = time_str_to_minutes(time_val)
        else:
            time_minutes = int(time_val)

        end_time_norm = normalize_time(time_minutes)
        features.append(end_time_norm)
        mask.append(1)
    else:
        features.append(0)
        mask.append(0)

    # Route
    if "route" in commute_data:
        route_encoding = encode_route(commute_data["route"])
        features.extend(route_encoding)
        mask.extend(np.ones(NUM_FEATURES["route"]))
    else:
        features.extend(np.zeros(NUM_FEATURES["route"]))
        mask.extend(np.zeros(NUM_FEATURES["route"]))

    # Stops
    if "stops" in commute_data:
        stops_encoding = encode_stops(commute_data["stops"])
        features.extend(stops_encoding)
        mask.extend(np.ones(NUM_FEATURES["stops"]))
    else:
        features.extend(np.zeros(NUM_FEATURES["stops"]))
        mask.extend(np.zeros(NUM_FEATURES["stops"]))

    return np.array(features, dtype=np.float32), np.array(mask, dtype=np.float32)


def batch_commute_to_features(commute_data_list: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert batch of commute data to feature matrices.

    Args:
        commute_data_list: List of commute data dictionaries

    Returns:
        Tuple of (features_matrix, masks_matrix) both of shape (n_samples, TOTAL_FEATURES)
    """
    features_list = []
    masks_list = []

    for commute in commute_data_list:
        features, mask = commute_to_features(commute)
        features_list.append(features)
        masks_list.append(mask)

    return (
        np.array(features_list, dtype=np.float32),
        np.array(masks_list, dtype=np.float32)
    )


def extract_predictions(features_vector: np.ndarray) -> Dict[str, Any]:
    """
    Extract human-readable information from predicted features vector.

    Args:
        features_vector: Predicted features vector of shape (TOTAL_FEATURES,)

    Returns:
        Dictionary with extracted features
    """
    idx = 0
    result = {}

    # Day of week (argmax of one-hot)
    day_encoding = features_vector[idx:idx + NUM_FEATURES["day_of_week"]]
    day_idx = int(np.argmax(day_encoding))
    result["dayOfWeek"] = list(DAY_MAPPING.keys())[day_idx]
    idx += NUM_FEATURES["day_of_week"]

    # Start time (denormalize)
    start_time_norm = features_vector[idx]
    start_time_minutes = int(start_time_norm * 1440)
    start_hours = start_time_minutes // 60
    start_mins = start_time_minutes % 60
    result["departureTime"] = f"{start_hours:02d}:{start_mins:02d}"
    idx += NUM_FEATURES["start_time"]

    # End time (denormalize)
    end_time_norm = features_vector[idx]
    end_time_minutes = int(end_time_norm * 1440)
    end_hours = end_time_minutes // 60
    end_mins = end_time_minutes % 60
    result["arrivalTime"] = f"{end_hours:02d}:{end_mins:02d}"
    idx += NUM_FEATURES["end_time"]

    # Route (argmax of one-hot, map back to name)
    route_encoding = features_vector[idx:idx + NUM_FEATURES["route"]]
    route_idx = int(np.argmax(route_encoding))
    route_names = list(ROUTE_MAPPING.keys())
    result["route"] = route_names[route_idx]
    idx += NUM_FEATURES["route"]

    # Stops (multi-hot, threshold 0.5)
    stops_encoding = features_vector[idx:idx + NUM_FEATURES["stops"]]
    stops = []
    if stops_encoding[0] > 0.5:
        stops.append("coffee")
    if stops_encoding[1] > 0.5:
        stops.append("lunch")
    if not stops:
        stops.append("none")
    result["stops"] = stops
    idx += NUM_FEATURES["stops"]

    return result
