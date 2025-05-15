import streamlit as st
import pandas as pd
from data_utils import load_csv, summarize_data
from llm_utils import generate_insights

st.title("Data Insights Generator")
st.write("Upload your CSV file")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = load_csv(uploaded_file)
    summary = summarize_data(df)
    st.subheader("Generating insights...")
    insights = generate_insights(summary, df)
    st.markdown(insights)