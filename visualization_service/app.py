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
API_GATEWAY_URL = os.getenv('WEB_MASTER_URL', os.getenv('STORAGE_URL', 'http://localhost:5001'))

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
        headers = {'Accept': 'application/json'}
        resp = requests.get(
            f'{API_GATEWAY_URL}/api/storage/dataset',
            params={'dataset_id': dataset_id, 'format': 'json'},
            headers=headers,
            timeout=10
        )
        if resp.status_code == 200:
            return pd.read_json(io.StringIO(resp.text), orient='split')
        return None
    except Exception as e:
        st.sidebar.error(f'Unable to get dataset with id {dataset_id}: {e}')
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
        fill='tozeroy',
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

st.sidebar.title('Панель управления')
is_healthy = check_gateway_health()
if is_healthy:
    st.sidebar.success('Web master is online!')
else:
    st.sidebar.error('Web master is offline!')

catalog = fetch_datasets_catalog()

if not catalog:
    st.title("📈 Visualization and data analysis service")
    st.info("No available datasets!")
    st.stop()

tab_bench, tab_distrib, tab_matrix = st.tabs([
    "🏆 Models benchmarks",
    "🔬 Dencity estimation",
    "🧩 Missings patterns"
])

with tab_bench:
    st.header("Effectiveness imputation algorithm comparison")
    df_benchmark = build_benchmarks_summary(catalog)
    if not df_benchmark.empty:
        st.dataframe(df_benchmark, use_container_width=True, hide_index=True)
        available_metrics = [c for c in df_benchmark.columns if c not in ["dataset_id", "dataset_name", "algorithm"]]

        if available_metrics:
            chosen_metric = st.selectbox("Choose metrics for diagramm:", available_metrics)
            fig_bar = plot_models_benchmark(df_benchmark, chosen_metric)
            st.plotly_chart(fig_bar, use_container_width=True)

            st.download_button(
                label=f"📥 Download ({chosen_metric}) in HTML",
                data=export_figure_to_html(fig_bar),
                file_name=f"benchmark_{chosen_metric}.html",
                mime="text/html"
            )

    else:
        st.warning("No available metrics!")

with tab_distrib:
    st.header("Estimation of dencity difference")
    orig_options = [f"{d['id']} | {d['name']}" for d in catalog if d.get("type") == "o"]
    imp_options = [f"{d['id']} | {d['name']}" for d in catalog if d.get("type") == "i"]

    col1, col2 = st.columns(2)
    with col1:
        sel_orig = st.selectbox("Original dataset", orig_options if orig_options else ['No data'])
    with col2:
        sel_imp = st.selectbox("Imputted dataset", imp_options if imp_options else ["No data"])

    if (sel_orig != 'No data') and (sel_imp != 'No data'):
        id_o = int(sel_orig.split(" | ")[0])
        id_i = int(sel_imp.split(" | ")[0])
        df_o = fetch_dataset_dataframe(id_o)
        df_i = fetch_dataset_dataframe(id_i)
        if df_o is not None and df_i is not None:
            common_num_cols = list(df_o.select_dtypes(include=[np.number]).columns.intersection(
                df_i.select_dtypes(include=[np.number]).columns
            ))

            if common_num_cols:
                feat = st.selectbox("Choose feature", common_num_cols)
                x_grid, kde_orig, kde_imp = calculate_kde_curves(df_o[feat], df_i[feat])

                if x_grid is not None:
                    fig_kde = plot_kde_comparison(x_grid, kde_orig, kde_imp, feat)
                    st.plotly_chart(fig_kde, use_container_width=True)

                    st.download_button(
                        label=f"📥 Download KDE-plot ({feat}) in HTML",
                        data=export_figure_to_html(fig_kde),
                        file_name=f"kde_{feat}.html",
                        mime="text/html"
                    )

                else:
                    st.warning('Choosen feature has zero variation!')

            else:
                st.info('No common numerical columns!')

with tab_matrix:
    st.header('Missings heatmap')
    sel_matrix_dataset = st.selectbox(
        "Choose dataset for missings analysis:",
        [f"{d['id']} | {d['name']}" for d in catalog]
    )
    if sel_matrix_dataset:
        d_id = int(sel_matrix_dataset.split(" | ")[0])
        df_mat = fetch_dataset_dataframe(d_id)
        if df_mat is not None and not df_mat.empty:
            total_cells = df_mat.size
            null_cells = int(df_mat.isnull().sum().sum())
            null_pct = (null_cells / total_cells) * 100 if total_cells > 0 else 0
            c1, c2 = st.columns(2)
            c1.metric("Total missings", f"{null_cells} cells")
            c2.metric("Fraction of missings", f"{null_pct:.2f}%")
            fig_mat = plot_missing_heatmap(df_mat)
            st.plotly_chart(fig_mat, use_container_width=True)


