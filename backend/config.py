"""Configuration schema and defaults for the ML model."""

DEFAULT_CONFIG = {
    # Training parameters
    "epochs": 30,
    "batch_size": 32,
    "learning_rate": 0.001,
    "train_val_split": 0.8,

    # Architecture
    "latent_dim": 8,

    # Loss weights
    "loss_weights": {
        "features": 1.0,
        "travel_time": 1.0,
        "pleasure": 0.5
    },

    # Inference parameters
    "inference_params": {
        "n_samples": 50,
        "time_weight": 1.0,
        "pleasure_weight": 1.0
    }
}

# Feature counts based on sandbox model
NUM_FEATURES = {
    "day_of_week": 7,      # One-hot encoded
    "start_time": 1,        # Normalized
    "end_time": 1,          # Normalized
    "route": 3,             # One-hot encoded
    "stops": 2,             # Multi-hot encoded
}

TOTAL_FEATURES = sum(NUM_FEATURES.values())


def validate_config(user_config: dict) -> dict:
    """
    Validate user-provided config and merge with defaults.

    Args:
        user_config: User-provided configuration overrides

    Returns:
        Validated configuration with defaults for missing values
    """
    if not isinstance(user_config, dict):
        raise ValueError("Config must be a dictionary")

    config = DEFAULT_CONFIG.copy()
    config.update(user_config)

    # Validate specific fields
    if config.get("epochs") and config["epochs"] <= 0:
        raise ValueError("epochs must be positive")

    if config.get("batch_size") and config["batch_size"] <= 0:
        raise ValueError("batch_size must be positive")

    if config.get("learning_rate") and config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")

    train_val_split = config.get("train_val_split", 0.8)
    if not (0 < train_val_split < 1):
        raise ValueError("train_val_split must be between 0 and 1")

    if config.get("latent_dim") and config["latent_dim"] <= 0:
        raise ValueError("latent_dim must be positive")

    return config
