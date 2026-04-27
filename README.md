---
title: Commutement
colorFrom: green
colorTo: green
sdk: docker
pinned: false
short_description: neural model for optimising commute for time and enjoyment
---

# commutement

a neural model for optimising daily commute for time and enjoyment

1. Flask backend for data collection, model training and prediction
2. React frontend for UI

## run

**terminal 1 - backend**

```bash
cd backend
pip install -r requirements.txt
flask --app server run
```

**terminal 2 - frontend**

```bash
cd frontend
npm install
npm run dev
```

## deploy

**backend**
deployed at huggingface spaces: <https://miafig-commutement.hf.space/>

**frontend**
deployed at github pages: <https://miafig.github.io/commutement/>

## model architecture

Use a masked conditional autoencoder with TensorFlow:

- **Input**: [day_of_week_one_hot(7), start_time(1), end_time(1), route_one_hot(7), sideQuests_multi_hot(7), ...]
- **Mask**: 1 = known, 0 = unknown for each feature
- **Encoder input**: concatenated [masked_features, mask]

**Encoder**: Input → Dense(64, relu) → Dense(32, relu) → latent_dim

**Decoder**: latent_dim → Dense(32, relu) → Dense(64, relu) → outputs:

- features_out (reconstructed features)
- travel_time_out (predicted travel time)
- pleasure_out (predicted pleasure)

The model uses a masked loss for feature reconstruction and predicts travel time and pleasure with linear output heads.

## TODO

- improve and test the model
- add UI updates for train and inference
