import pandas as pd
import numpy as np
import plotly.express as px
def load_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    df = df.dropna()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['name'] = df['name'].str.replace('&amp;', '&')
    df['chg_%'] = df['chg_%'].str.replace('%', '')
    df['chg_%'] = df['chg_%'].astype('float')
    df['vol_'] = df['vol_'].apply(parse_volume)

    return df

def parse_volume(vol):
    vol = vol.replace('M', 'e6').replace('K', 'e3')
    return float(eval(vol))

def max_price(company, df):
    return df[df['name'] == company]['high'].max()

def min_price(company, df):
    return df[df['name'] == company]['low'].min()

def avg_price(company, df):
    return df[df['name'] == company]['close'].mean()

def total_volume(company, df):
    return df[df['name'] == company]['vol_'].sum()

def get_companies(df):
    return df['name'].unique().tolist()

def get_company_stats(df, company):
    company_data = df[df['name'] == company]

    max_price = company_data['high'].max()
    min_price = company_data['low'].min()    
    avg_price = company_data['last'].mean()
    total_volume = company_data['vol_'].sum()

    return max_price, min_price, avg_price, total_volume

