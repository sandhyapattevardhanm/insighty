import pandas as pd
import numpy as np

def clean_dict(d):
    """Recursively remove NaN/inf from dictionaries."""
    if isinstance(d, dict):
        return {k: clean_dict(v) for k, v in d.items()}
    elif isinstance(d, float) and (np.isnan(d) or np.isinf(d)):
        return None
    return d

def clean_column_names(df):
    """Remove empty/strange column names."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and c != ":"]]
    return df

def perform_basic_analysis(df):
    df = clean_column_names(df)
    df = df.dropna(axis=1, how='all')
    
    for col in df.select_dtypes(include='number').columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

    numeric_corr = df.select_dtypes(include='number').corr().to_dict()
    numeric_corr = clean_dict(numeric_corr)

    numeric_corr = {
        k: {kk: numeric_corr[k].get(kk) for kk in list(numeric_corr[k].keys())[:2]}
        for k in list(numeric_corr.keys())[:2]
    }

    group_analysis = {}
    if 'gender' in df.columns:
        gender_group = df.groupby('gender').mean(numeric_only=True)
        group_analysis['gender_group_stats'] = clean_dict(gender_group.to_dict())

    return {
        "correlations": numeric_corr,
        "group_analysis": group_analysis
    }

def summarize_data(df):
    df = clean_column_names(df)
    cols_to_show = [c for c in df.columns.tolist() if c.strip()][:15]  # Limit columns in prompt
    summary = {
        "columns": cols_to_show,
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "missing_values": clean_dict(df.isnull().sum().to_dict()),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample_rows": df.head(3).to_dict(orient="index")
    }
    return summary
