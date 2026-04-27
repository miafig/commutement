"""Data processing and feature engineering for commute data."""

import numpy as np
from typing import Tuple, Dict, Any, List
from config import NUM_FEATURES


# Mapping for day names to indices
DAY_MAPPING = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6,
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

# Route name to ID mapping
ROUTE_MAPPING = {
    "": 0,
    "paddington": 1,
    "monument": 2,
    "monument+walk": 3,
    "monument+circle": 4,
    "circle+hsk+walk": 5,
    "picadilly+central": 6,
    "other": 0
}

# Stop name to ID mapping
SIDE_QUESTS_MAPPING = {
    "none": -1,
    "drink": 0,
    "breakfast": 1,
    "lunch": 2,
    "walk": 3,
    "grocery": 4,
    "shop": 5,
    "errand": 6,
}

# Disruption name to ID mapping
DISRUPTION_MAPPING = {
    "none": -1,
    "strikes": 0,
    "delay":1,
    "holiday": 2,
    "late": 3,
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
    encoding[route_id] = 1.0
    return encoding


def encode_side_quests(side_quests_data: Any) -> np.ndarray:
    """
    Multi-hot encode side quests.

    Args:
        side_quests_data: Side quest info (can be list like ["drink", "walk"] or a single side quest string)

    Returns:
        Multi-hot encoded vector of length 7
    """
    encoding = np.zeros(NUM_FEATURES["sideQuests"])

    def add_side_quest(side_quest_name: str) -> None:
        side_quest_key = side_quest_name.lower().strip()
        if side_quest_key == "none" or side_quest_key == "":
            return
        side_quest_id = SIDE_QUESTS_MAPPING.get(side_quest_key, -1)
        if side_quest_id >= 0:
            encoding[side_quest_id] = 1.0

    if isinstance(side_quests_data, list):
        for side_quest_name in side_quests_data:
            if isinstance(side_quest_name, str):
                add_side_quest(side_quest_name)
    elif isinstance(side_quests_data, str):
        add_side_quest(side_quests_data)

    return encoding

def encode_disruptions(disruption: str) -> np.ndarray:
    """
    One-hot encode disruption.

    Args:
        disruption: Disruption name (e.g., 'strikes')

    Returns:
        One-hot encoded vector of length 4
    """
    encoding = np.zeros(NUM_FEATURES["disruptions"])

    def add_disruption(disruption_name: str) -> None:
        disruption_key = disruption_name.lower().strip()
        if disruption_key == "none" or disruption_key == "":
            return
        disruption_id = DISRUPTION_MAPPING.get(disruption_key, -1)
        if disruption_id >= 0:
            encoding[disruption_id] = 1.0

    if isinstance(disruption, list):
        for disruption_name in disruption:
            if isinstance(disruption_name, str):
                add_disruption(disruption_name)
    elif isinstance(disruption, str):
        add_disruption(disruption)
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
        "sideQuests": ["coffee"],
        "pleasureRating": 5,
        ...
    }

    Returns:
        Tuple of (features_vector, mask_vector) both of shape (TOTAL_FEATURES,)
    """
    features = []
    mask = []

    # Day of week
    if "dayOfWeek" in commute_data:
        day_encoding = encode_day_of_week(commute_data["dayOfWeek"])
        features.extend(day_encoding)
        mask.extend(np.ones(NUM_FEATURES["day_of_week"]))
    else:
        features.extend(np.zeros(NUM_FEATURES["day_of_week"]))
        mask.extend(np.zeros(NUM_FEATURES["day_of_week"]))

    # Going
    if 'going' in commute_data:
        going_val = commute_data['going']
        if isinstance(going_val, str):
            going_val = going_val.lower() in ['work', 'home']
        features.append(1 if going_val == 'work' else 0)
        mask.append(1)
    else:
        features.append(0.0)
        mask.append(0)

    # Start time (departureTime)
    if "departureTime" in commute_data:
        time_val = commute_data["departureTime"]

        if isinstance(time_val, str):
            time_minutes = time_str_to_minutes(time_val)
        else:
            time_minutes = int(time_val)

        start_time_norm = normalize_time(time_minutes)
        features.append(start_time_norm)
        mask.append(1)
    else:
        features.append(0.0)
        mask.append(0)

    # End time (arrivalTime)
    if "arrivalTime" in commute_data:
        time_val = commute_data["arrivalTime"]

        if isinstance(time_val, str):
            time_minutes = time_str_to_minutes(time_val)
        else:
            time_minutes = int(time_val)

        end_time_norm = normalize_time(time_minutes)
        features.append(end_time_norm)
        mask.append(1)
    else:
        features.append(0.0)
        mask.append(0)

    # Transport
    if "transport" in commute_data:
        transport_val = commute_data["transport"].lower().strip()
        transport_encoding = 1.0 if transport_val == "train" else 0.0
        features.append(transport_encoding)
        mask.append(1)
    else:
        features.append(0.0)
        mask.append(0)

    # Route
    if "route" in commute_data:
        route_encoding = encode_route(commute_data["route"])
        features.extend(route_encoding)
        mask.extend(np.ones(NUM_FEATURES["route"]))
    else:
        features.extend(np.zeros(NUM_FEATURES["route"]))
        mask.extend(np.zeros(NUM_FEATURES["route"]))

    # Side Quests
    if "sideQuests" in commute_data:
        side_quests_encoding = encode_side_quests(commute_data["sideQuests"])
        features.extend(side_quests_encoding)
        mask.extend(np.ones(NUM_FEATURES["sideQuests"]))
    else:
        features.extend(np.zeros(NUM_FEATURES["sideQuests"]))
        mask.extend(np.zeros(NUM_FEATURES["sideQuests"]))

    # Disruptions
    if "disruptions" in commute_data:
        disruptions_encoding = encode_disruptions(commute_data["disruptions"])
        features.extend(disruptions_encoding)
        mask.extend(np.ones(NUM_FEATURES["disruptions"]))
    else:
        features.extend(np.zeros(NUM_FEATURES["disruptions"]))
        mask.extend(np.zeros(NUM_FEATURES["disruptions"]))
    
    # Company
    if "company" in commute_data:
        company_val = commute_data["company"].lower().strip()
        company_encoding = 1.0 if company_val == "yes" else 0.0
        features.append(company_encoding)
        mask.append(1)
    else:
        features.append(0.0)
        mask.append(0)
    
    # Rush
    if "rush" in commute_data:
        rush_val = commute_data["rush"].lower().strip()
        rush_encoding = 2.0 if rush_val == "high" else 1.0 if rush_val == "medium" else 0.0
        features.append(rush_encoding)
        mask.append(1)
    else:
        features.append(0.0)
        mask.append(0)

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

    # Going
    going_val = features_vector[idx]
    result["going"] = "work" if going_val >= 0.5 else "home"
    idx += NUM_FEATURES["going"]

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

    # Transport
    transport_val = features_vector[idx]
    result["transport"] = "train" if transport_val >= 0.5 else "bike"
    idx += NUM_FEATURES["transport"]

    # Route (argmax of one-hot)
    route_encoding = features_vector[idx:idx + NUM_FEATURES["route"]]
    route_idx = int(np.argmax(route_encoding))
    route_names = list(ROUTE_MAPPING.keys())
    result["route"] = route_names[route_idx]
    idx += NUM_FEATURES["route"]

    # Side Quests (multi-hot)
    side_quests_encoding = features_vector[idx:idx + NUM_FEATURES["sideQuests"]]
    side_quests = []
    reverse_side_quests = {v: k for k, v in SIDE_QUESTS_MAPPING.items() if v >= 0}
    for quest_idx, value in enumerate(side_quests_encoding):
        if value > 0.5:
            side_quests.append(reverse_side_quests.get(quest_idx, "unknown"))
    result["sideQuests"] = side_quests
    idx += NUM_FEATURES["sideQuests"]

    # Disruptions (multi-hot)
    disruptions_encoding = features_vector[idx:idx + NUM_FEATURES["disruptions"]]
    disruptions = []
    reverse_disruptions = {v: k for k, v in DISRUPTION_MAPPING.items() if v >= 0}
    for disruption_idx, value in enumerate(disruptions_encoding):
        if value > 0.5:
            disruptions.append(reverse_disruptions.get(disruption_idx, "unknown"))
    result["disruptions"] = disruptions
    idx += NUM_FEATURES["disruptions"]

    # Company
    company_val = features_vector[idx]
    result["company"] = "yes" if company_val >= 0.5 else "no"
    idx += NUM_FEATURES["company"]

    # Rush
    rush_val = round(float(features_vector[idx]))
    rush_val = max(0, min(rush_val, 2))
    result["rush"] = "high" if rush_val == 2 else "medium" if rush_val == 1 else "low"
    idx += NUM_FEATURES["rush"]

    return result
