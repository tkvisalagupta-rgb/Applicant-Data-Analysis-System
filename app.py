import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from candidate_pipeline import (
    load_data,
    clean_data,
    add_scores,
    recompute_scores_with_weights,
    get_top_candidates,
)

st.set_page_config(page_title="Applicant Data Analysis System", layout="wide")

st.title("Applicant Data Analysis System")
st.write("Analyze applicant data and rank candidates for a Data Analyst role.")

st.sidebar.header("Data Source")

use_default = st.sidebar.checkbox(
    "Use default dataset from ./data/applicants.csv", value=True
)

uploaded_file = None
df_raw = None

if use_default:
    try:
        df_raw = load_data("data/applicants.csv")
        st.sidebar.success("Loaded data/applicants.csv")
    except FileNotFoundError:
        st.sidebar.error("data/applicants.csv not found. Upload a CSV instead.")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload your applicant CSV file", type=["csv"]
    )
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        st.sidebar.success("Uploaded custom CSV")

if df_raw is None:
    st.warning("Please upload a CSV file or place 'applicants.csv' in the data folder.")
    st.stop()

df_clean = clean_data(df_raw)
df_scored = add_scores(df_clean)

st.subheader("Data Preview (with scores)")
st.dataframe(df_scored.head())

st.sidebar.header("Scoring Weights")

w_test = st.sidebar.slider("Test Score Weight", 0.0, 1.0, 0.4, 0.05)
w_interview = st.sidebar.slider("Interview Score Weight", 0.0, 1.0, 0.3, 0.05)
w_exp = st.sidebar.slider("Experience Weight", 0.0, 1.0, 0.2, 0.05)
w_tier = st.sidebar.slider("Company Tier Weight", 0.0, 1.0, 0.1, 0.05)

w_sum = w_test + w_interview + w_exp + w_tier
if w_sum == 0:
    w_test, w_interview, w_exp, w_tier = 0.4, 0.3, 0.2, 0.1
    w_sum = 1.0

w_test /= w_sum
w_interview /= w_sum
w_exp /= w_sum
w_tier /= w_sum

df_scored = recompute_scores_with_weights(
    df_scored, w_test, w_interview, w_exp, w_tier
)

st.sidebar.write("Normalized Weights:")
st.sidebar.write(
    {
        "Test": round(w_test, 2),
        "Interview": round(w_interview, 2),
        "Experience": round(w_exp, 2),
        "Tier": round(w_tier, 2),
    }
)

st.subheader("Top Candidates")

top_n = st.slider("Number of candidates to display:", min_value=3, max_value=20, value=5)

top_df = get_top_candidates(df_scored, n=top_n)

cols_to_show = [
    col
    for col in [
        "Name",
        "Degree",
        "Years of Experience",
        "Test Score",
        "Interview Score",
        "Past Company Tier",
        "Location Preference",
        "final_score",
    ]
    if col in top_df.columns
]

st.dataframe(top_df[cols_to_show])

st.subheader("Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    if "Years of Experience" in df_scored.columns and "Test Score" in df_scored.columns:
        st.markdown("**Experience vs Test Score**")
        fig1, ax1 = plt.subplots()
        sns.scatterplot(
            data=df_scored,
            x="Years of Experience",
            y="Test Score",
            ax=ax1,
        )
        ax1.set_xlabel("Years of Experience")
        ax1.set_ylabel("Test Score")
        st.pyplot(fig1)

with col2:
    if "Degree" in df_scored.columns:
        st.markdown("**Degree Distribution**")
        fig2, ax2 = plt.subplots()
        df_scored["Degree"].value_counts().plot(kind="bar", ax=ax2)
        ax2.set_xlabel("Degree")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)

st.markdown("You can adjust the weights on the left and see how rankings change.")
