Insighty – AI Data Insights Generator

Insighty is a Streamlit app that turns any CSV dataset into business-friendly insights.
It combines Pandas preprocessing with Generative AI (FLAN-T5) for natural language analysis, with a Pandas fallback to ensure results even if AI fails.

🚀 Features
Upload CSV → Clean & summarize data
Generate actionable insights with LLM
Automatic fallback insights without AI
Quick stats: missing values, correlations, top categories

🛠 Tech Stack
Python, Streamlit, Pandas, Hugging Face Transformers, PyTorch

⚡ Run Locally
pip install -r requirements.txt
streamlit run app.py

📌 Example Output
Drama is the most common genre, making up 40% of titles.
English-language films have an average IMDb score of 6.8, higher than non-English films.
Runtime and IMDb score show a weak positive correlation (0.12).
