import streamlit as st


st.title("📈 Model Performance Analysis")

st.write("""
This page compares the performance of different machine learning models
used for student placement prediction.
""")
import pandas as pd

df = pd.read_csv("report/model_comparison.csv")

st.subheader("📊 Model Comparison")
st.dataframe(df, width='content') 

st.subheader("ROC-AUC Comparison")

st.markdown("""
- Logistic Regression Tuned : 0.881
- Random Forest Tuned : 0.873
""")

from PIL import Image
image=Image.open("images/roc_auc_lr.png")
st.image(image,width='content')

st.success("""
🏆 Final Selected Model: Logistic Regression Tuned 

Reasons:
- Highest ROC-AUC Score (0.881)
- Better Recall for placed students
- Lower False Negatives (162)
- Good generalization performance
- Simpler and more interpretable than Random Forest Tuned 
""")
