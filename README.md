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
flask run
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

- **Input**: [day_of_week_one_hot(7), start_time, end_time, route_one_hot(5), stops_multi_hot(5), ...]
- **Mask**: 1=known, 0=unknown for each feature

**Encoder**: Input → Dense(64, relu) → Dense(32, relu) → latent_dim

**Decoder**: latent_dim → Dense(32, relu) → Dense(64, relu) → outputs:

- features_out (reconstructed features)
- time_out (predicted arrival time)
- pleasure_out (subjective enjoyment, sigmoid)

## TODO

- fix train and inference APIs
- improve and test the model
- model /predict should not predict variables different from the ones given eg. if an input is Monday then it shouldn't predict Friday
