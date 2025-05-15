import nest_asyncio
nest_asyncio.apply()

# Followed by other imports
import asyncio
import torch
import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load FLAN-T5 model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Function to clean dictionary (remove NaN or infinite values)
def clean_dict(d):
    if isinstance(d, dict):
        return {k: clean_dict(v) for k, v in d.items()}
    elif isinstance(d, float) and (np.isnan(d) or np.isinf(d)):
        return None
    return d

# Function to perform basic analysis on the dataset
def perform_basic_analysis(df):
    df = df.dropna(axis=1, how='all')
    df = df.copy()
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

# Function to summarize the dataset
def summarize_data(df):
    summary = {
        "columns": df.columns.tolist()[:10],
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.apply(lambda dt: str(dt)).to_dict(),
        "sample_rows": df.head(3).to_dict(orient="index"),
        "column_stats": df.describe(include='all').to_dict(),
        "value_counts": {
            col: df[col].value_counts().to_dict()
            for col in df.select_dtypes(include='object').columns
        }
    }
    return summary

# Function to generate insights using the model
def generate_insights(summary, df):
    analysis = perform_basic_analysis(df)
    sample_rows = list(summary['sample_rows'].values())[:3]

    prompt = f"""
Dataset contains {summary['num_rows']} rows and {summary['num_columns']} columns.
Columns: {summary['columns']}
Missing values: {summary['missing_values']}
Numeric correlations: {analysis['correlations']}
Group analysis: {analysis['group_analysis']}
Sample: {sample_rows}

Write 4-5 insights in business language.
"""

    output = generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.8)
    return output[0]['generated_text']

# Function to load CSV file
def load_csv(file_path):
    return pd.read_csv(file_path)

# Main function to test the model with a CSV file
def test_model_with_csv(file_path):
    df = load_csv(file_path)
    summary = summarize_data(df)
    insights = generate_insights(summary, df)
    print("Generated Insights:")
    print(insights)
