import pandas as pd

def load_csv(file):
    df = pd.read_csv(file)
    return df

def summarize_data(df):
    summary = {
        "columns": df.columns.tolist(),
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.apply(lambda dt: str(dt)).to_dict(),
        "sample_rows": df.head(5).to_dict(orient="index")
    }
    return summary