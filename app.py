import streamlit as st
import pandas as pd
from data_utils import summarize_data
from llm_utils import generate_insights

st.title("📊 Data Insights Generator")
st.write("Upload your CSV file to generate business insights using AI.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="main_csv_uploader")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    summary = summarize_data(df)

    st.subheader("Dataset Overview")
    st.write(f"**Rows:** {summary['num_rows']}")
    st.write(f"**Columns:** {summary['num_columns']}")
    st.write(f"**First Columns:** {summary['columns']}")

    st.subheader("Generating insights...")
    insights = generate_insights(summary, df)
    st.markdown(insights)
