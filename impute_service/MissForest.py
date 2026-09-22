import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder 



class MissForest:
   def __init__(self, n_iterations = 10, n_estimators = 100, random_state = 42, categories = []):
      self.iterations = n_iterations
      self.estimators = n_estimators
      self.random_state = random_state
      self.categories = categories
      self.encoders = {}
      self.original_missing = None  
   
   def fit_transform(self, X):
      if isinstance(X,pd.DataFrame):
         self.feature_names = X.columns.tolist()
         X = X.copy()
      else:
         X = pd.DataFrame(X)
      
      X_encoded = X.copy()
      self.original_missing = X.isna().copy()  
      
      for col in self.categories:
        le  = LabelEncoder()
        
        all_vals = set(X.iloc[:, col].dropna())
        le.fit(list(all_vals))
        encoded_col = X.iloc[:,col].map(lambda x: le.transform([x])[0] if pd.notna(x) else np.nan)
        X_encoded[X.columns[col]] = encoded_col.astype(float) 
        self.encoders[col] = le

      X_imp = X_encoded.copy()
      n_features = X_imp.shape[1]
      
      # Начальная импутация
      for col in range(n_features):
         mask = X_imp.iloc[:,col].isna()
         if X_imp.iloc[:,col].isna().any():
            if col in self.categories:
               mode_val = X_imp.iloc[:,col].mode()
               if not mode_val.empty:
                  fill_value = int(mode_val.iloc[0])
                  X_imp.iloc[mask,col] = fill_value
               else:
                  X_imp.iloc[mask,col] = 0 
            else:
               median_val = X_imp.iloc[:,col].median()
               mask = X_imp.iloc[:,col].isna()
               if pd.isna(median_val):  
                  median_val = 0
               X_imp.iloc[mask,col] = median_val
      
      for iteration in range(self.iterations):
         X_old = X_imp.copy()

         for col in range(n_features):
            missing_mask = self.original_missing.iloc[:, col]  
            if not missing_mask.any():
               continue
            
            other_cols = [c for c in range(n_features) if c!=col]

            train_X = X_imp.iloc[~missing_mask.values, other_cols]
            train_y = X_imp.iloc[~missing_mask.values, col]

            test_X = X_imp.iloc[missing_mask.values, other_cols]
            
            if col in self.categories:
               train_y = train_y.astype(int)  # <-- ИСПРАВЛЕНИЕ 6: преобразование в int
               model = RandomForestClassifier(n_estimators = self.estimators, random_state = self.random_state, n_jobs = -1)
            else:
               model = RandomForestRegressor(n_estimators = self.estimators, random_state = self.random_state, n_jobs = -1)
            
            model.fit(train_X,train_y)
            predictions = model.predict(test_X)
            X_imp.iloc[missing_mask.values, col] = predictions
         #if np.allclose(X_imp.values, X_old.values,equal_nan = True):
           # break
         
      result = X_imp.copy()
      for col, le in self.encoders.items():
         col_data = result.iloc[:, col]
         col_data = pd.to_numeric(col_data, errors='coerce').fillna(0).astype(int)
         
         # ИСПРАВЛЕНИЕ: Заменяем столбец целиком через имя
         col_name = result.columns[col]
         result[col_name] = le.inverse_transform(col_data)

      if self.feature_names is not None:
         result = pd.DataFrame(result,columns=self.feature_names)
      return result
   