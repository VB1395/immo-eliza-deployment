# 🏡 Real Estate Price Prediction - Immo Eliza - deployment

## 🏢 Description

Welcome to the immo-eliza-deployment repository where we have deployed a machine learning model that predicts real estate prices through an API endpoint. This solution consists of a FastAPI backend service deployed on Render and a frontend web application using Streamlit.

## 📦 Repo structure
```.
├── api/
│ |── app.py
| ├── appCstBoost.py
| ├── predict.py
├── data/
│ ├── houses.csv
├── model/
│ |── CatBoost.pkl
| ├── RandomForest_model.pkl
| ├── encoding.pkl
├── stremlit/
│ |── app.py
| ├── predict.py
├── .gitignore
├── Dockerfile
├── predict.py
├── README.md
└── requirements.txt
```