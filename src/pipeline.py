import pandas as pd
import numpy as np

from src.utils import * 
from src.feature_engineering import *
from src.run import *

def basic_data_pipeline():
    df = get_train_data()
    df = reduce_mem_usage(df)
    df = drop_useless_columns(df)
    df = drop_constant_columns(df)
    df = type_conversion(df)
    df, rest_datetime_cols = get_rest_datetime_columns(df)
    df = fix_datetime_columns(df, rest_datetime_cols)
    df = searchRoute(df)
    df = frequentFlyer(df)
    return df

def fe1_pipeline(df):
    df = fe_columns(df)
    df = top_company(df)
    df = cabin_class_features(df)
    df = is_direct_flight(df)
    return df