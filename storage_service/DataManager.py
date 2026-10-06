import sqlite3
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
import os
import json
import uuid


@dataclass
class DatasetRow:
    original_name: str
    table_class: str
    file_path: str
    missing_rate: float
    missing_count: int
    target_only_missing: int
    dataset_status: str
    n_cols: int
    n_rows: int
    target_column: str = None
    created_at: Optional[str] = None
    imputation_method: Optional[str] = None
    dataset_id: Optional[int] = None

    def __post_init__(self):
        if self.table_class not in ['o', 'i', 'd']:
            raise ValueError('Undefined table class!')
        if self.dataset_status not in ['n', 'c', 'm']:
            raise ValueError('Undefined dataset status!')
        if not (0 <=self.missing_rate <= 1):
            raise ValueError('Missing rate is out of range [0;1]!')
        if self.target_only_missing not in [0, 1]:
            raise ValueError('Undefined target only status!')

class DatasetManager:

    def __init__(self, db_path, table_name='datasets'):
        self.db_path = db_path
        self.table_name = table_name
        self.conn = sqlite3.connect(database=db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type:
            self.conn.rollback()
        else:
            self.conn.commit()
        self.conn.close()

    def __getitem__(self, id):
        if isinstance(id, int):
            res = self.cursor.execute(f'SELECT * FROM {self.table_name} WHERE dataset_id = ?', (id,)).fetchone()
            if not res:
                raise KeyError(f'Dataset with {id} not found!')
            return dict(res)
        elif isinstance(id, str):
            res = self.cursor.execute(f"SELECT * FROM {self.table_name} WHERE original_name = ? AND table_class = 'o' AND missing_rate = 0", (id,)).fetchone()
            if not res:
                raise KeyError(f'Dataset with name {id} not found!')
            return dict(res)
        elif isinstance(id, tuple):
            res = self.cursor.execute(f'SELECT * FROM {self.table_name} WHERE original_name = ? AND table_class = ? AND ABS(missing_rate - ?) < 0.001', id).fetchone()
            if not res:
                raise KeyError(f'Dataset with name {id} not found!')
            return dict(res)

    def __delitem__(self, id):
        res = False
        if isinstance(id, int):
            res = self.delete_by_id(id)
        elif isinstance(id, str):
            res = self.delete_by_data(id)
        elif isinstance(id, tuple):
            res = self.delete_by_data(*id)
        else:
            raise TypeError(f"Invalid key type: {type(id)}")
        if not res:
            raise KeyError(f"Dataset with key {id} not found!")
        return res


    def add(self, row:DatasetRow):
        script = f'''INSERT INTO {self.table_name}
        (original_name,
        table_class,
        file_path,
        missing_rate,
        missing_count,
        target_only_missing,
        dataset_status,
        n_cols,
        n_rows,
        imputation_method,
        target_column) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''

        row.dataset_id = self.cursor.execute(script,(
        row.original_name,
        row.table_class,
        row.file_path,
        row.missing_rate,
        row.missing_count,
        row.target_only_missing,
        row.dataset_status,
        row.n_cols,
        row.n_rows,
        row.imputation_method,
        row.target_column)).lastrowid
        self.conn.commit()
        return row.dataset_id

    def give(self, row:DatasetRow):
        script = f'''SELECT dataset_id from {self.table_name}
        WHERE original_name = ? AND
        table_class = ? AND
        missing_rate = ? AND
        target_only_missing = ? AND
        dataset_status = ? AND
        imputation_method IS ?
        '''
        id = self.cursor.execute(script, (
            row.original_name,
            row.table_class,
            row.missing_rate,
            row.target_only_missing,
            row.dataset_status,
            row.imputation_method
        )).fetchone()
        if id:
            return self[id['dataset_id']]
        else:
            return False

    def __len__(self):
        script = f'SELECT COUNT(*) FROM {self.table_name}'
        return self.conn.execute(script).fetchone()[0]

    def delete_by_id(self, id:int):
        try:
            path_data = self[id]['file_path']
        except KeyError:
            return False
        path_json = os.path.splitext(path_data)[0] + '.json' if path_data else None
        self.cursor.execute(f'DELETE FROM {self.table_name} WHERE dataset_id = ?', (id,))
        self.conn.commit()
        if path_data and os.path.exists(path_data):
            try:
                os.remove(path_data)
            except OSError:
                pass
        if path_json and os.path.exists(path_json):
                    try:
                        os.remove(path_json)
                    except OSError:
                        pass
        return True

    def delete_by_data(self,name, status='o', fraction=0.0):
        id = self.cursor.execute(f'SELECT dataset_id FROM {self.table_name} WHERE original_name = ? AND ABS(missing_rate - ?) < 0.001 AND table_class = ?', (name, fraction, status)).fetchone()
        if not id:
            return False
        else:
            return self.delete_by_id(id['dataset_id'])



@dataclass
class MetricsRow:
    dataset_id: int
    accuracy_weighted: Optional[float] = None
    precision_weighted: Optional[float] = None
    recall_weighted: Optional[float] = None
    f1_score_weighted: Optional[float] = None
    accuracy_macro: Optional[float] = None
    precision_macro: Optional[float] = None
    recall_macro: Optional[float] = None
    f1_score_macro: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    created_at: Optional[str] = None

class MetricsManager:

    def __init__(self, db_path, table_name='metrics'):
        self.db_path = db_path
        self.table_name = table_name
        self.conn = sqlite3.connect(database=db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type:
            self.conn.rollback()
        self.conn.close()

    def __getitem__(self, id):
            res = self.cursor.execute(f'SELECT * FROM {self.table_name} WHERE dataset_id = ?', (id,)).fetchone()
            if not res:
                raise KeyError(f'Dataset with {id} not found!')
            return dict(res)

    def add(self, row:MetricsRow):
        script = f'''INSERT OR REPLACE INTO {self.table_name}
        (dataset_id,
        accuracy_weighted,
        precision_weighted,
        recall_weighted,
        f1_score_weighted,
        accuracy_macro,
        precision_macro,
        recall_macro,
        f1_score_macro,
        mse,
        mae) VALUES (?,?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''

        self.cursor.execute(script,(
        row.dataset_id,
        row.accuracy_weighted,
        row.precision_weighted,
        row.recall_weighted,
        row.f1_score_weighted,
        row.accuracy_macro,
        row.precision_macro,
        row.recall_macro,
        row.f1_score_macro,
        row.mse,
        row.mae))
        self.conn.commit()
        return row.dataset_id

    def __len__(self):
        script = f'SELECT COUNT(*) FROM {self.table_name}'
        return self.conn.execute(script).fetchone()[0]

class StorageService:

    def __init__(self, db_path, storage_path):
        self.db_path = os.path.abspath(db_path)
        self.storage_path = os.path.abspath(storage_path)

    def register_data(self, config, data):
        json_data = json.load(config)
        df = pd.read_csv(data)
        config.seek(0)
        data.seek(0)

        original_name = json_data.get('original_name', getattr(data, 'filename', 'dataset.csv'))
        
        table_class = json_data.get('table_class') or json_data.get('status', 'o')
        dataset_status = json_data.get('dataset_status') or json_data.get('status', 'o')
        imputation_method = json_data.get('imputation_method') or json_data.get('imputer', None)
        missing_rate = json_data.get('missing_rate', json_data.get('fraction', 0.0))

        match table_class:
            case 'o': subdir = 'original'
            case 'i': subdir = 'imputed'
            case 'd' | 'c': subdir = 'dirty'
            case _: subdir = 'other'

        file_uuid = uuid.uuid4().hex
        base_path = os.path.join(self.storage_path, subdir)
        os.makedirs(name=base_path, exist_ok=True)

        json_path = os.path.join(base_path, f'{file_uuid}.json')
        csv_path = os.path.join(base_path, f'{file_uuid}.csv')
        data.save(csv_path)
        config.save(json_path)

        n_rows = len(df)
        n_cols = len(df.columns)
        file_path = csv_path
        missing_count = int(df.isna().sum().sum())
        target_column = json_data.get('target_column', None)
        target_only_missing = 0
        if target_column and target_column in df.columns and (missing_count == df[target_column].isna().sum()):
            target_only_missing = 1

        row = DatasetRow(
            original_name=original_name, 
            table_class=table_class, 
            file_path=file_path, 
            missing_rate=float(missing_rate),
            missing_count=missing_count, 
            target_only_missing=target_only_missing, 
            dataset_status=dataset_status,
            n_cols=n_cols, 
            n_rows=n_rows, 
            target_column=target_column,
            imputation_method=imputation_method
        )

        with DatasetManager(self.db_path, table_name='datasets') as d:
            row_id = d.add(row)

        return row_id

