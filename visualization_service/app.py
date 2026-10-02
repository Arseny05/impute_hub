import streamlit as st
import os
import io
from dotenv import load_dotenv
import json
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import requests 
import pandas as pd
import numpy as np

load_dotenv()
API_GATEWAY_URL = os.getenv('WEB_MASTER_PORT', os.getenv('STORAGE_URL', 'http://localhost:5001'))

def check_gateway_health():
    try:
        resp = requests.get(f'{API_GATEWAY_URL}/api/storage/dataset', timeout=5)
        return resp.status_code in [200, 404]
    except Exception:
        return False

def fetch_datasets_catalog():
    try:
        resp = requests.get(f'{API_GATEWAY_URL}/api/storage/dataset', timeout=5)
        if resp.status_code == 200:
            return resp.json().get('datasets', [])
        return []
    except Exception as e:
        st.sidebar.error(f'Error occured while getting catalog {e}')
        return []

def fetch_dataset_dataframe(dataset_id):
    try:
        resp = requests.get(f'{API_GATEWAY_URL}/api/storage/dataset/dataset_id', timeout=10)
        if resp.status_code == 200:
            return pd.read_json(io.StringIO(resp.text), orient='split')
        return None

    except Exception as e:
        st.sidebar.error(f'Unable to get dataset with id {dataset_id}')
        return None

def fetch_imputation_metrics(dataset_id):
    try:
        resp = requests.get(f"{API_GATEWAY_URL}/api/storage/metrics/{dataset_id}", timeout=5)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


def calculate_kde_curves(series_orig, series_imp, num_points=200):
    vals_orig = series_orig.dropna().values
    vals_imp = series_imp.dropna().values
    if np.var(vals_orig) < 1e-9 or np.var(vals_imp) < 1e-9:
        return None, None, None
    
    x_min = min(np.min(vals_orig), np.min(vals_imp))
    x_max = max(np.max(vals_orig), np.max(vals_imp))
    x_grid = np.linspace(x_min, x_max, num_points)

    kde_orig_estimator = stats.gaussian_kde(vals_orig)
    kde_imp_estimator = stats.gaussian_kde(vals_imp)
    kde_orig = kde_orig_estimator(x_grid)
    kde_imp = kde_imp_estimator(x_grid)
    return x_grid, kde_orig, kde_imp

def build_benchmarks_summary(datasets_list):
    records = []
    for item in datasets_list:
        if item.get('type') == 'i':
            m_data = fetch_imputation_metrics(item["id"])
            if m_data and "metrics" in m_data:
                record = {
                    "dataset_id": item["id"],
                    "dataset_name": item.get("name", f"Dataset {item['id']}"),
                    "algorithm": item.get("algorithm", "Undefined"),
                    **m_data["metrics"]
                }
                records.append(record)
    return pd.DataFrame(records)

def plot_kde_comparison(x_grid, kde_orig, kde_imp, feature_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_grid,
        y=kde_orig,
        mode="lines",
        name="Original",
        line=dict(color="#1f77b4", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(31, 119, 180, 0.15)"
    ))
    fig.add_trace(go.Scatter(
        x=x_grid,
        y=kde_imp,
        mode='lines',
        name='Imputted',
        line=dict(color="#e75b03", width=2.5, dash='dash'),
        fill='tozeroy'
        fillcolor="rgba(255, 127, 14, 0.15)"
    ))

    fig.update_layout(
        title=f"Kernel density estimation: feature '{feature_name}'",
        xaxis_title=feature_name,
        yaxis_title="Probability density",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0.75, y=0.98, bgcolor="rgba(255,255,255,0.7)")
    )
    return fig

def plot_models_benchmark(df_bench, metric_col):
    fig = px.bar(df_bench,
                x='algorithm',
                y=metric_col,
                color="algorithm",
                text_auto=".4f",
                title=f"Comparison algorithms by metric {metric_col.upper()}",
                template="plotly_white")
    fig.update_layout(showlegend=False, xaxis_title="Algorithm", yaxis_title=metric_col.upper())
    return fig

def plot_missing_heatmap(df, max_rows=500):
    missing_sub = df.iloc[:max_rows].isnull().astype(int)
    fig = px.imshow(
        missing_sub.T,
        labels=dict(x="Index string", y="Feature", color="Missing"),
        color_continuous_scale=[[0, "#2ca02c"], [1, "#d62728"]],
        title=f"Heatmap missings (showed first {min(max_rows, len(df))} rows)"
    )
    fig.update_layout(coloraxis_showscale=False, template="plotly_white")
    return fig

def export_figure_to_html(fig):
    return fig.to_html(include_plotlyjs="cdn").encode("utf-8")


st.set_page_config(page_title="Visualization Service", page_icon="📈", layout="wide")

pass
