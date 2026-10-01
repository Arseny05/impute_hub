import pandas as pd
import numpy as np
import requests
import streamlit as st
import plotly.express as px
import seaborn as sns 
import dotenv
import os
import pickle
from Arim import Arim
from MissForest import MissForest
from Mode import impute_by_mode
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, accuracy_score
import sys
import matplotlib.pyplot as plt
import io
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
ENV_PATH = os.path.join(ROOT_DIR, ".env")

dotenv.load_dotenv(dotenv_path=ENV_PATH)

ML_URL = os.getenv("ML_URL", 'http://localhost:5003')
STORAGE_URL = os.getenv("STORAGE_URL", "http://storage_service:5001")

def evaluate_classification(original, imputed, missed, regime='macro'):
    if regime not in ['macro', 'weighted', 'micro']:
        print('Incorrect classification regime!')
        return None
    missing_positions = missed.isnull()
    y_true = []
    y_pred = []
    results = {}
    results['by_column'] = {}
    for col in original.columns:
        col_true = []
        col_pred = []
        for idx in original.index:
            if pd.isna(missed.at[idx, col]):
                y_true.append(str(original.at[idx, col]))
                y_pred.append(str(imputed.at[idx, col]))
                col_true.append(str(original.at[idx, col]))
                col_pred.append(str(imputed.at[idx, col]))
        if col_true:
            results['by_column'][col] = {'imputed_count': len(col_true), 'accuracy': accuracy_score(col_true, col_pred)}
    results['total_imputed'] =  len(y_true)
    results['accuracy'] = accuracy_score(y_true, y_pred) if y_true else 0.0
    results['precision'] = precision_score(y_true, y_pred, average=regime, zero_division=0.0) if y_true else 0.0
    results['recall'] = recall_score(y_true, y_pred, average=regime, zero_division=0.0) if y_true else 0.0
    results['f1_score'] = f1_score(y_true, y_pred, average=regime, zero_division=0.0) if y_true else 0.0
    return results 

def evaluate_regression(original, imputed, missed):
    missing_positions = missed.isnull()
    y_true = []
    y_pred = []
    results = {}
    results['by_column'] = {}
    for col in original.columns:
        col_true = []
        col_pred = []
        for idx in original.index:
            if pd.isna(missed.at[idx, col]):
                try:
                    y_true.append(float(original.at[idx, col]))
                    y_pred.append(float(imputed.at[idx, col]))
                    col_true.append(float(original.at[idx, col]))
                    col_pred.append(float(imputed.at[idx, col]))
                except Exception:
                    pass
        if col_true:
            results['by_column'][col] = {'imputed_count': len(col_true), 'MSE': mean_squared_error(col_true, col_pred), 'MAE': mean_absolute_error(col_true, col_pred)}
    results['total_imputed'] =  len(y_true)
    results['MSE'] = mean_squared_error(y_true, y_pred) if y_true else 0.0
    results['MAE'] = mean_absolute_error(y_true, y_pred) if y_true else 0.0
    return results

def resout(original,imputed,missed, num_cols, regime='macro'):
    if num_cols:
        results_reg = evaluate_regression(original[num_cols], imputed[num_cols], missed[num_cols])
        st.subheader("📈 Metrics by numerical columns:")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total imputted", results_reg['total_imputed'])
        m2.metric("Total MSE", f"{results_reg['MSE']:.4f}")
        m3.metric("Total MAE", f"{results_reg['MAE']:.4f}")
        if results_reg['by_column']:
            df_reg = pd.DataFrame.from_dict(results_reg['by_column'], orient='index')
            df_reg.index.name = "Column"
            st.dataframe(
                df_reg.style.format({
                    "imputed_count": "{:.0f}",
                    "MSE": "{:.4f}",
                    "MAE": "{:.4f}"
                }),
                use_container_width=True
            )
    if set(num_cols) != set(imputed.columns):
        cat_cols = list(set(imputed.columns) - set(num_cols))
        results_clf = evaluate_classification(original[cat_cols], imputed[cat_cols], missed[cat_cols], regime=regime)
        if results_clf:
            st.subheader("🏷️ Metrics by categorical columns:")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{results_clf['accuracy']:.4f}")
            c2.metric("Precision", f"{results_clf['precision']:.4f}")
            c3.metric("Recall", f"{results_clf['recall']:.4f}")
            c4.metric("F1-score", f"{results_clf['f1_score']:.4f}")
            if results_clf['by_column']:
                df_clf = pd.DataFrame.from_dict(results_clf['by_column'], orient='index')
                df_clf.index.name = "Column"
                st.dataframe(
                    df_clf,
                    column_config={
                        "imputed_count": st.column_config.NumberColumn("Imputted", format="%d"),
                        "accuracy": st.column_config.ProgressColumn(
                            "Accuracy",
                            min_value=0.0,
                            max_value=1.0,
                            format="%.4f"
                        )
                    },
                    use_container_width=True
                )

def plot_metrics(original, imputed, missed, num_cols, regime='macro'):
    st.divider()
    st.header("📊 Plots of imputation metrics")

    if num_cols:
        reg_res = evaluate_regression(original[num_cols], imputed[num_cols], missed[num_cols])
        if reg_res and reg_res['by_column']:
            st.subheader("Numerical columns metrics")
            cols = list(reg_res['by_column'].keys())
            mse_vals = [reg_res['by_column'][c]['MSE'] for c in cols]
            mae_vals = [reg_res['by_column'][c]['MAE'] for c in cols]

            fig_reg, ax = plt.subplots(figsize=(8, 4))

            ax.plot(cols, mse_vals, marker='o', label='MSE', color='#4C72B0')
            ax.plot(cols, mae_vals, marker='s', label='MAE', color='#55A868')
            ax.set_ylabel('Error value')
            ax.set_title('Imputation errors by columns (MSE / MAE)')
            ax.set_xticklabels(cols, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig_reg)

            buf_reg = io.BytesIO()
            fig_reg.savefig(buf_reg, format='png', dpi=300)
            st.download_button(label='📥 Download regression plot (PNG)', data=buf_reg.getvalue(), file_name='regression_metrics.png', mime='image/png')
            plt.close(fig_reg)

        cat_cols = list(set(imputed.columns) - set(num_cols))
        if cat_cols:
            clf_res = evaluate_classification(original[cat_cols], imputed[cat_cols], missed[cat_cols], regime=regime)
            if clf_res and clf_res['by_column']:
                st.subheader("Categorial columns accuracy")
                cols = list(clf_res['by_column'].keys())
                acc_vals = [clf_res['by_column'][c]['accuracy'] for c in cols]
                fig_cat, ax = plt.subplots(figsize=(9, 4.5))
                ax.plot(cols, acc_vals, marker='o', linewidth=2, color='#C44E52', label='Accuracy')
                ax.set_ylabel('Accuracy')
                ax.set_ylim([0.0, 1.05])
                ax.set_title('Accuracy by columns')
                ax.set_xticklabels(cols, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                st.pyplot(fig_cat)

                buf_cat = io.BytesIO()
                fig_cat.savefig(buf_cat, format='png', dpi=300)
                st.download_button(label='📥 Download accuracy plot (PNG)', data=buf_cat.getvalue(), file_name='accuracy_metrics.png', mime='image/png')
                plt.close()

def get_dataset(param:dict):
    param['as_file'] = True
    endpoint = f"{STORAGE_URL}/api/storage/dataset"
    try:
        with st.spinner('Loading data...'):
            response = requests.get(endpoint, params=param, timeout=15)
        if response.status_code == 200:
            payload = pickle.loads(response.content)
            return payload['data'], payload['metadata']
        else:
                try:
                    err_msg = response.json().get("error", response.text)
                except Exception:
                    err_msg = response.text
                st.error(f"Error ({response.status_code}): {err_msg}")
                return None, None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to storage service: {e}")
        return None, None

def upload_imputed_dataset(df_imputed, original_meta, model_name):
    endpoint = f"{STORAGE_URL}/api/storage/upload"
    orig_name = original_meta.get("original_name", "dataset") if original_meta else "dataset"
    fraction = original_meta.get("fraction", 0.0) if original_meta else 0.0
    csv_buffer = io.BytesIO()
    df_imputed.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    config_dict = {
        "original_name": orig_name,
        "fraction": fraction,
        "status": "i", 
        "imputer": model_name
    }
    config_buffer = io.BytesIO(json.dumps(config_dict).encode("utf-8"))   
    files = {
        "file": (f"{orig_name}_imputed_{model_name.lower()}.csv", csv_buffer, "text/csv"),
        "config": ("config.json", config_buffer, "application/json")
    }
    try:
        response = requests.post(endpoint, files=files, timeout=20)
        if response.status_code == 201:
            dataset_id = response.json().get("dataset_id")
            return dataset_id
        else:
            try:
                err_msg = response.json().get("error", response.text)
            except Exception:
                err_msg = response.text
            st.error(f"Error while loading dataset ({response.status_code}): {err_msg}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Unable to connect storage!: {e}")
        return None

def save_metrics_to_storage(dataset_id, original, imputed, missed, num_cols):
    endpoint = f"{STORAGE_URL}/api/storage/metrics"
    payload = {"dataset_id": int(dataset_id)}
    if num_cols:
        reg_res = evaluate_regression(original[num_cols], imputed[num_cols], missed[num_cols])
        if reg_res:
            payload["mse"] = float(reg_res.get("MSE", 0.0))
            payload["mae"] = float(reg_res.get("MAE", 0.0))
    cat_cols = list(set(imputed.columns) - set(num_cols))
    if cat_cols:
        clf_macro = evaluate_classification(original[cat_cols], imputed[cat_cols], missed[cat_cols], regime="macro")
        if clf_macro:
            payload["accuracy_macro"] = float(clf_macro.get("accuracy", 0.0))
            payload["precision_macro"] = float(clf_macro.get("precision", 0.0))
            payload["recall_macro"] = float(clf_macro.get("recall", 0.0))
            payload["f1_score_macro"] = float(clf_macro.get("f1_score", 0.0))
        clf_weighted = evaluate_classification(original[cat_cols], imputed[cat_cols], missed[cat_cols], regime="weighted")
        
        if clf_weighted:
            payload["accuracy_weighted"] = float(clf_weighted.get("accuracy", 0.0))
            payload["precision_weighted"] = float(clf_weighted.get("precision", 0.0))
            payload["recall_weighted"] = float(clf_weighted.get("recall", 0.0))
            payload["f1_score_weighted"] = float(clf_weighted.get("f1_score", 0.0))

    try:
        response = requests.post(endpoint, json=payload, timeout=15)
        if response.status_code == 201:
            return True
        else:
            try:
                err_msg = response.json().get("error", response.text)
            except Exception:
                err_msg = response.text
            st.error(f"Error saving metrics ({response.status_code}): {err_msg}")
            return False
    except requests.exceptions.RequestException as e:
        st.error(f"Unable to send metrics: {e}")
        return False


if __name__ == '__main__':

    st.set_page_config(page_title="Impute Service", layout='wide')
    st.title('Missing data imputer')
    if "current_df" not in st.session_state:
        st.session_state["current_df"] = None
    if "df_metadata" not in st.session_state:
        st.session_state["df_metadata"] = None
    if "original_df" not in st.session_state:
        st.session_state["original_df"] = None
    if "imputed_df" not in st.session_state:
        st.session_state["imputed_df"] = None

    st.header('1. Choose dataset:')

    query_type = st.radio('Choose type of getting dataset:', options=['By ID', 'By parameters (name, missings rate, status)'])

    if query_type == 'By ID':
        dataset_id = st.number_input(label='dataset_id...', min_value=1, step=1)
        if st.button('Load by ID'):
            df, meta = get_dataset(param={'dataset_id':dataset_id})
            if df is not None:
                st.session_state['imputed_df'] = None
                st.session_state['current_df'] = df
                st.session_state['df_metadata'] = meta
                name = meta.get('original_name')
                st.session_state['original_df'], _ = get_dataset(param={'name': name, 'fraction': 0.0, 'status': 'o'})
    else:
        param = {}
        param['name'] = st.text_input("Input name of dataset:", 'dataset')
        param['fraction'] = st.slider(label='Choose rate of missings...', min_value=0.0, max_value=0.95, step=0.05, value=0.2)
        param['status'] = st.selectbox('Choose dataset status:', ['o', 'i', 'd'])
        if st.button(label = 'Load by params'):
            df, meta = get_dataset(param = param)
            if df is not None:
                st.session_state['imputed_df'] = None
                st.session_state['current_df'] = df
                st.session_state['df_metadata'] = meta
                st.session_state['original_df'], _ = get_dataset(param={'name': param['name'], 'fraction': 0.0, 'status': 'o'})

    if st.session_state['current_df'] is not None:
        df = st.session_state['current_df']
        col_view1, col_view2 = st.columns([3, 1])
        with col_view1:
            st.write("Preview first 10 strings:")
            st.dataframe(df.head(10), use_container_width=True)

        with col_view2:
            st.write("Missings by columns:")
            missing_counts = df.isnull().sum()
            st.dataframe(missing_counts[missing_counts > 0], use_container_width=True)
        num_cols = st.multiselect(label='Choose numerical columns:', options=df.columns, default=[])


        st.divider()
        st.header('2. Choosing imputer:')
        model = st.selectbox(label="Choose imputer:", options=['Arim', 'MissForest', 'Mode'])
        if model == 'Arim':
            min_supp = st.slider('Choose min_supp', min_value=0.0, max_value=1.0, step=0.01, value=0.01)
            min_conf = st.slider('Choose min_conf', min_value=0.0, max_value=1.0, step=0.01, value=0.5)
            max_rules = st.number_input(label='Max_rules...', min_value=500, step=500, value=None)
            n_bins = st.number_input(label='N bins...', min_value=1,  max_value=20, step=1, value=5)
            obj = Arim(num_cols=num_cols, min_supp=min_supp, min_conf=min_conf, max_rules=max_rules, n_bins=n_bins)
            if st.button('Start Arim'):
                with st.spinner('Arim fitting...'):
                    obj.fit(st.session_state['current_df'])
                with st.spinner('Arim imputting...'):
                    imputed = obj.transform(st.session_state['current_df'])
                st.success('Arim has imputted missing data succesfully!')
                st.session_state['imputed_df'] = imputed
        if model == 'MissForest':
            obj = MissForest(categories=st.session_state['current_df'].columns.get_indexer(list(set(st.session_state['current_df'])-set(num_cols))))
            if st.button('Start MissForest'):
                with st.spinner('MissForest fitting...'):
                    obj.fit(st.session_state['current_df'])
                with st.spinner('MissForest imputting...'):
                    imputed = obj.transform(st.session_state['current_df'])
                st.success('MissForest has imputted missing data succesfully!')
                st.session_state['imputed_df'] = imputed
        if model == 'Mode':
            if st.button('Start Mode'):
                with st.spinner('Mode imputting'):
                    imputed = impute_by_mode(st.session_state['current_df'])
                st.success('Mode has imputted missing data succesfully')
                st.session_state['imputed_df'] = imputed

        if st.session_state['imputed_df'] is not None and st.session_state['original_df'] is not None:
            st.divider()
            st.header('3. Metrics:')
            regime = st.radio(label='Choose estimation regime:', options=['weighted', 'macro', 'micro'])
            resout(
                original=st.session_state['original_df'],
                imputed=st.session_state['imputed_df'],
                missed=st.session_state['current_df'],
                num_cols=num_cols,
                regime=regime
            )
            plot_metrics(original=st.session_state['original_df'],
                        imputed=st.session_state['imputed_df'],
                        missed=st.session_state['current_df'],
                        num_cols=num_cols,
                        regime=regime)
            st.divider()
            if st.button('Save on storage'):
                with st.spinner("Saving imputted dataset on storage..."):
                    new_id = upload_imputed_dataset(
                        df_imputed=st.session_state['imputed_df'],
                        original_meta=st.session_state.get('df_metadata'),
                    model_name=model
                    )
                if new_id is not None:
                    st.success(f'Dataset succesfully registered with id {new_id}')
                    if st.session_state['original_df'] is not None:
                        with st.spinner("Calculating metrics and saving them in the storage..."):
                            metrics_ok = save_metrics_to_storage(
                            dataset_id=new_id,
                            original=st.session_state['original_df'],
                            imputed=st.session_state['imputed_df'],
                            missed=st.session_state['current_df'],
                            num_cols=num_cols
                        )
                        if metrics_ok:
                            st.success('Metrics succesfully has been written on the storage!')
                    else: 
                        st.warning('Original dataset has not been found! Metrics have not been calculated!')

    st.divider()
    col_reset, _ = st.columns([1, 4])
    with col_reset:
        if st.button('Drop all and begin again'):
            st.session_state.clear()
            st.rerun()
            
           

        
    

    
        
            
            

