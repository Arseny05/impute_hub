import pandas as pd
import numpy as np

def impute_by_mode(imputed):
    mode = imputed.mode().iloc[0]
    return imputed.fillna(mode)

