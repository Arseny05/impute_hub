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
        results = evaluate_regression(original[num_cols], imputed[num_cols], missed[num_cols])
        print('Numerical columns:')
        print(f'Total imputed: {results['total_imputed']}')
        print(f'MSE: {results['MSE']}')
        print(f'MAE: {results['MAE']}')
        for key, value in results['by_column'].items():
            print(f'{key}:')
            print(f'Imputed count: {results['by_column'][key]['imputed_count']}')
            print(f'MSE: {results['by_column'][key]['MSE']}')
            print(f'MAE: {results['by_column'][key]['MAE']}')
            print('-'*35)
    if set(num_cols) != set(imputed.columns):
        cat_cols = list(set(imputed.columns) - set(num_cols))
        results = evaluate_classification(original[cat_cols], imputed[cat_cols], missed[cat_cols], regime=regime)
        print('Categorical columns:')
        print(f'Total imputed: {results['total_imputed']}')
        print(f'Accuracy: {results['accuracy']}')
        print(f'Precision: {results['precision']}')
        print(f'Recall: {results['recall']}')
        print(f'F1-score: {results['f1_score']}')
        for key, value in results['by_column'].items():
            print(f'{key}:')
            print(f'Imputed count: {results['by_column'][key]['imputed_count']}')
            print(f'Accuracy: {results['by_column'][key]['accuracy']}')
            print('-'*35)

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
            pass
        if model == 'Mode':
            pass
        
    

    
        
            
            

