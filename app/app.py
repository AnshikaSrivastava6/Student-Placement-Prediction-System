import streamlit as st

st.set_page_config(
    page_title="Student Placement Prediction System",
    page_icon="🎓"
)

st.title("🎓 Student Placement Prediction System")

st.markdown("""
Welcome to the Student Placement Prediction System.

Use the sidebar to navigate through the application.
""")
st.markdown("""
## About the Project

This project predicts whether a student is likely to be placed based on:

- Academic Performance
- Internships
- Projects
- Certifications
- Aptitude Skills
- Soft Skills
- Extracurricular Activities

The prediction is generated using a Machine Learning model trained on student placement data.
""")

st.markdown("---")

st.subheader("🚀Workflow")

st.markdown("""
1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Selection
5. Model Training
6. Model Evaluation
7. Deployment using FastAPI and Streamlit
""")

st.markdown("---")

st.subheader("📊 Dataset Overview")

st.markdown("""
- Total Records: 9,928
- Target Variable: Placement Status
- Class Distribution:
    - Not Placed: 5,801
    - Placed: 4,1927
- Features Used: 10
""")

st.subheader("🚀 Technologies Used")

st.markdown("""
- Python
- Pandas
- Scikit-Learn
- Matplotlib
- Seaborn
- Streamlit
- FastAPI
- Joblib
""")