import pandas as pd
import streamlit as st
import numpy as np
import requests
from dotenv import load_dotenv 
import os
import io
import json
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
ENV_PATH = os.path.join(ROOT_DIR, ".env")

load_dotenv(dotenv_path=ENV_PATH)

STORAGE_URL = os.getenv('STORAGE_URL', 'http://localhost:5001')

st.set_page_config(page_title="Collector Service", layout='wide')
st.title("Data Collector")

if "corrupted_columns" not in st.session_state:
    st.session_state["corrupted_columns"] = []

if "df_raw" not in st.session_state:
    st.session_state["df_raw"] = None
if "df_current" not in st.session_state:
    st.session_state["df_current"] = None
if "is_corrupted" not in st.session_state:
    st.session_state["is_corrupted"] = False
if "missing_rate" not in st.session_state:
    st.session_state["missing_rate"] = 0.0

st.header('1. Data loading:')
source_type = st.radio('Choose a type of loading', ['URL', 'Local file'])
if source_type == 'URL':
    url = st.text_input("URL of dataset (CSV)...", 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
    if st.button("Download by URL"):
        try:
            df = pd.read_csv(url)
            st.session_state['df_raw'] = df
            st.session_state["df_current"] = df.copy()
            st.session_state["is_corrupted"] = False
            st.success(f"Succesfully loaded {len(df)} strings, {df.shape[1]} tables")
        except Exception as e:
            st.error(str(e))
else:
    uploaded_file = st.file_uploader("Choose CSV-file...", type=['csv'])
    if uploaded_file is not None and st.session_state.get('last_file') != uploaded_file.name:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['df_raw'] = df
            st.session_state["df_current"] = df.copy()
            st.session_state["is_corrupted"] = False
            st.session_state['last_file'] = uploaded_file.name
            st.success(f"Succesfully loaded {len(df)} strings, {df.shape[1]} tables")
        except Exception as e:
            st.error(str(e))

if st.session_state['df_current'] is not None:
    df = st.session_state['df_current']
    st.divider()
    st.header("2. Data analysis")
    col_view1, col_view2 = st.columns([3, 1])
    with col_view1:
        st.write("Preview first 10 strings:")
        st.dataframe(df.head(10), use_container_width=True)

    with col_view2:
        st.write("Missings by columns:")
        missing_counts = df.isnull().sum()
        st.dataframe(missing_counts[missing_counts > 0], use_container_width=True)

    st.divider()
    cols_corrupt = []
    percent = 0
    st.header('3. Making missings in dataset')
    strategy = st.selectbox(
        "Choose the columns with missings:",
        options=["All columns", "Target column only", "Dataset should not have any missings"]
    )
    if strategy == 'All columns':
        cols_corrupt = list(df.columns)
        st.info("Missings will be evenly distributed along the table")
    elif strategy == 'Target column only':
        target_col = st.selectbox('Select target column: ', options=list(df.columns))
        st.info(f'{target_col} will be the only column with missings')
        cols_corrupt = [target_col]
    else:
        cols_corrupt = []
    if cols_corrupt:
        percent = st.slider('Chose percentage of missings:', 5, 95, 20)
        col_btn1, col_btn2 = st.columns([1,4])
        df_corrupted = st.session_state['df_raw'].copy()
        rate = percent/100
        with col_btn1:
            if st.button('Make missings'):
                mask = np.random.rand(*df_corrupted[cols_corrupt].shape) <= rate
                df_corrupted[cols_corrupt] = df_corrupted[cols_corrupt].mask(mask)
                st.session_state["df_current"] = df_corrupted
                st.session_state["is_corrupted"] = True
                st.session_state["missing_rate"] = rate
                st.session_state["corrupted_columns"] = cols_corrupt
                st.success(f"Made {percent}% missings to table!")
                time.sleep(2)
                st.rerun()
        with col_btn2:
            if st.button('Return raw dataset'):
                st.session_state["df_current"] = st.session_state["df_raw"].copy()
                st.session_state["is_corrupted"] = False
                st.session_state["missing_rate"] = 0.0
                st.session_state["corrupted_columns"] = []
                st.success('Table returned into raw state')
                time.sleep(2)
                st.rerun()
st.divider()
st.header('4. Register into dataset storage')
if st.session_state['df_current'] is not None:
    dataset_name = st.text_input("Dataset name:", "dataset_sample")
    dataset_status = st.selectbox('Choose dataset status:', ['n', 'c', 'm'])
    if st.button('Send dataset', type='primary'):
        config = {
            'name':dataset_name,
            'table_class': 'd' if st.session_state['is_corrupted'] else 'o', 
            'missing_rate': st.session_state['missing_rate'],
            'target_column': st.session_state.get('corrupted_columns', [None])[0] if len(st.session_state.get('corrupted_columns', [])) == 1 else None,
            'dataset_status': dataset_status
        }
        csv_buffer = io.StringIO()
        st.session_state['df_current'].to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")
        config_bytes = json.dumps(config, ensure_ascii=False, indent=2).encode("utf-8")
        files = {
            "file":(f"{dataset_name}.csv", csv_bytes, "text/csv"),
            "config": (f"{dataset_name}.json", config_bytes, "application/json")
        }
        endpoint = f"{STORAGE_URL}/api/storage/upload"
        try:
            with st.spinner(f"Sending data on {endpoint}..."):
                response = requests.post(endpoint, files=files,timeout=15)
            if response.status_code in [200, 201]:
                res_data = response.json()
                st.success("Датасет успешно сохранен в хранилище!")
                st.json(res_data)
            else:
                    st.error(f"Server returned code: {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
                st.error(f"Failed to contact with Storage Service; adress: {endpoint}: {e}")   










