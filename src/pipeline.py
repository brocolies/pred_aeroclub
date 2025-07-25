import pandas as pd
import numpy as np

from src.utils import * 
from src.feature_engineering import *
from src.run import *

def pipeline():
    df = get_train_data()
    df = reduce_mem_usage(df)
    df = drop_constant_columns(df)
    df = type_conversion(df)
    df, rest_datetime_cols = get_rest_datetime_columns(df)
    df = fix_datetime_columns(df, rest_datetime_cols)
    df = searchRoute(df)
    df = frequentFlyer(df)
    
    return df