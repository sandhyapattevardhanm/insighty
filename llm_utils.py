from data_utils import perform_basic_analysis
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load model once
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

generator = load_model()

def basic_fallback_insights(df):
    """Generate non-LLM fallback insights."""
    insights = []
    if "Singer" in df.columns:
        top_singers = df["Singer"].value_counts().head(5).to_dict()
        insights.append(f"Top 5 most common singers: {top_singers}")
    
    if {"energy", "valence"}.issubset(df.columns):
        avg_energy = df.groupby("Singer")["energy"].mean().sort_values(ascending=False).head(1)
        happiest_song = df.loc[df["valence"].idxmax(), "Song name"] if "Song name" in df.columns else None
        insights.append(f"Singer with highest average energy: {avg_energy.index[0]} ({avg_energy.values[0]:.2f})")
        if happiest_song:
            insights.append(f"Happiest song in dataset: {happiest_song} (valence: {df['valence'].max():.2f})")

    if {"energy", "valence"}.issubset(df.columns):
        corr = df["energy"].corr(df["valence"])
        insights.append(f"Correlation between energy and valence: {corr:.2f}")

    return "\n".join(insights)

def generate_insights(summary, df):
    """Super basic but always works for demo."""
    try:
        insights = []

        # Most common value for each categorical column
        for col in df.select_dtypes(include="object").columns[:3]:
            top_val = df[col].mode()[0]
            count = df[col].value_counts().iloc[0]
            insights.append(f"The most common {col} is '{top_val}' appearing {count} times.")

        # Average of first few numeric columns
        for col in df.select_dtypes(include="number").columns[:3]:
            avg_val = df[col].mean()
            insights.append(f"The average {col} is {avg_val:.2f}.")

        # Correlation between first two numeric columns
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) >= 2:
            corr_val = df[num_cols[0]].corr(df[num_cols[1]])
            insights.append(f"The correlation between {num_cols[0]} and {num_cols[1]} is {corr_val:.2f}.")

        return "\n".join(insights)

    except Exception as e:
        return f"⚠️ Error generating insights: {str(e)}"
