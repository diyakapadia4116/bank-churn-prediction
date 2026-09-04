"""Shared helpers for the Bank Churn Risk Dashboard.

Feature engineering here mirrors the notebook exactly so that a row built
from user input is scored consistently with training.
"""
import json
import joblib
import pandas as pd
import streamlit as st

EPS = 1.0


@st.cache_resource
def load_pipeline(path="models/churn_pipeline.joblib"):
    return joblib.load(path)


@st.cache_data
def load_metadata(path="models/metadata.json"):
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_leaderboard(path="models/model_leaderboard.csv"):
    return pd.read_csv(path)


@st.cache_data
def load_feature_importance(path="models/feature_importance.csv"):
    return pd.read_csv(path)


@st.cache_data
def load_population_scores(path="models/population_scores.csv"):
    return pd.read_csv(path)


@st.cache_data
def load_raw_data(path="data/European_Bank.csv"):
    return pd.read_csv(path)


def build_feature_row(CreditScore, Geography, Gender, Age, Tenure, Balance,
                       NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary,
                       feature_columns):
    """Build a single-row DataFrame with engineered features, column-aligned
    to the order the pipeline was trained on."""
    row = pd.DataFrame([{
        'CreditScore': CreditScore, 'Geography': Geography, 'Gender': Gender,
        'Age': Age, 'Tenure': Tenure, 'Balance': Balance,
        'NumOfProducts': NumOfProducts, 'HasCrCard': HasCrCard,
        'IsActiveMember': IsActiveMember, 'EstimatedSalary': EstimatedSalary
    }])
    row['BalanceSalaryRatio'] = row['Balance'] / (row['EstimatedSalary'] + EPS)
    row['ProductDensity'] = row['NumOfProducts'] / (row['Tenure'] + EPS)
    row['EngagementProductInteraction'] = row['IsActiveMember'] * row['NumOfProducts']
    row['AgeTenureInteraction'] = row['Age'] * row['Tenure']
    row['ZeroBalanceFlag'] = (row['Balance'] == 0).astype(int)
    return row[feature_columns]


def risk_tier(probability):
    if probability >= 0.60:
        return "High Risk", "#C94C4C"
    elif probability >= 0.30:
        return "Medium Risk", "#E0A73B"
    else:
        return "Low Risk", "#1B8A5A"
