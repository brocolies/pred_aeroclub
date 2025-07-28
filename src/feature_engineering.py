import pandas as pd
import numpy as np

def drop_constant_columns(df):
    col_names = df.columns.tolist()
    for col in col_names:
        if df[col].nunique() == 1:
            df = df.drop(columns=f'{col}')
    return df

def drop_useless_columns(df):
    segments3_cols = []
    entire_col_names = df.columns.tolist()
    for col in entire_col_names:
        if 'segments3' in col:
            segments3_cols.append(col)
    df = df.drop(columns=segments3_cols)
    return df

def type_conversion(df):
    for col in df.columns:
        if 'duration' in col:
            # duration_split = df[col].str.split(':')
            # hour = duration_split.str[0].astype(int)
            # minute = duration_split.str[1].astype(int)
            clean_duration = df[col].str.replace(r'(\d+)\.0+:', r'\1:', regex=True)
            df[f'{col}_minutes'] = pd.to_timedelta(clean_duration,
  errors='coerce').dt.total_seconds() / 60
            # df[f'{col}_minutes'] = pd.to_timedelta(df[col]).dt.total_seconds() / 60
            df = df.drop(columns=[col])

        elif df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                df[col] = df[col].astype('category')

        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_hour'] = df[col].dt.hour
            df = df.drop(columns=[col])
    return df 

def get_rest_datetime_columns(df):
    col_names = df.columns.tolist()
    rest_datetime_cols = []
    for col in col_names:
        if 'arrivalAt' in col or 'departureAt' in col:
            rest_datetime_cols.append(col)
    
    return df, rest_datetime_cols
    
def fix_datetime_columns(df, rest_datetime_cols):
    for col in rest_datetime_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)

        # datetime 피처 생성
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_day'] = df[col].dt.day
        df[f'{col}_hour'] = df[col].dt.hour
        df[f'{col}_dayofweek'] = df[col].dt.dayofweek
    df = df.drop(columns=rest_datetime_cols)
    return df

def searchRoute(df):
    df[['route_1', 'route_2']] = df['searchRoute'].str.split('/', expand=True)
    df['departure_from'] = df['route_1'].str[:3]
    df['departure_to'] = df['route_1'].str[3:]
    df['return_from'] = df['route_2'].str[:3]
    df['return_to'] = df['route_2'].str[3:]
    df = df.drop(columns=['route_1', 'route_2'])

    return df

def frequentFlyer(df):
    df['frequentFlyer_count'] = df['frequentFlyer'].str.split('/').str.len().fillna(0).astype('int8')
    df['is_frequentFlyer'] = df['frequentFlyer'].str.len() > 0

    return df

def fe_columns(df):
    df['ff_company_matches'] = df.apply(
    lambda row: str(row['legs0_segments0_operatingCarrier_code']) in str(row['frequentFlyer']), 
    axis=1
    )
    
    return df

def top_company(df):
    # 이용률이 높은 항공사 위주로 is_ feature 생성
    top_company = df['legs0_segments0_operatingCarrier_code'].value_counts().head(30)
    top_company = top_company > 100000
    top_company = top_company[top_company].index.tolist()
    for company in top_company:
        df[f'is_{company}'] = (
            df['legs0_segments0_operatingCarrier_code'] == company).astype(int)
    df = df.drop(columns='frequentFlyer')
    return df

def is_direct_flight(df):
    df['is_direct_flight'] = df['legs0_segments1_aircraft_code'].isnull().astype(int)
    return df

# def baggage_measurement_type(df):
#     legs = [0, 1]
#     segments = [0, 1, 2]
#     for i in legs:
#         for j in segments:
#             weight_col = f'legs{i}_segments{j}_baggageAllowance_weightMeasurementType'
#             quantity_col = f'legs{i}_segments{j}_baggageAllowance_quantity'

#             # 컬럼 존재 여부 확인
#             if weight_col in df.columns and quantity_col in df.columns:
#                 df[f'legs{i}_segments{j}_is_weight_based'] = (df[weight_col] == 1) & (df[quantity_col] > 3)
#                 df[f'legs{i}_segments{j}_is_no_baggage'] = (df[weight_col] == 1) & (df[quantity_col] == 0)
#     return df

def cabin_class_features(df):
    cabin_mapping = {1: 1, 2: 2, 3: 4, 4: 3}

    for i in [0, 1]:
        for j in [0, 1, 2]:
            col = f'legs{i}_segments{j}_cabinClass'
            if col in df.columns:
                df[col] = df[col].map(cabin_mapping)
    return df