import streamlit as st
from PIL import Image

st.title("📊Exploratory Data Analysis")
st.markdown("""
           This page presents important Visualizations and Insights from Student Placement Data""")

st.header("Placement Status Distribution")
image=Image.open("images/placement_count.png")
st.image(image,width='content')
st.markdown("""
       INSIGHT : 
    - Non-placed students are slightly more than placed student
    - The target variable is moderately balanced.
    - No significant class imbalance is present.
    - Hence, SMOTE was not applied during model training.""")
st.markdown("----")
st.subheader("Categorical Analysis")
image=Image.open("images/categorical_analysis.png")
st.image(image,width='content')
st.markdown("""
       INSIGHT :
    - Extracurricular activities show a positive relationship with placement status.
    - Placement training has a strong positive impact on placement outcomes.
    - Students involved in both activities generally demonstrate better placement performance.""")

st.markdown("----")
st.subheader("Numerical Analysis")
image=Image.open("images/numerical_analysis.png")
st.image(image,width='content')
st.markdown("""
       INSIGHT :
    - Higher CGPA is associated with better placement outcomes.
    - Students with more internships tend to have higher placement rates.
    - Project experience shows a positive relationship with placement status.
    - Workshops and certifications contribute positively to employability.""")

st.markdown("----")
st.subheader("Skills Analysis")
image=Image.open("images/skill_analysis.png")
st.image(image,width='content')
st.markdown("""
       INSIGHT :
    - Students with Higher SSC_Marks tend to have higher placement rates .
    - Students with more HSC_Marks tend to have higher placement rates.
    - Aptitude Test Score is an important indicator of placement success.
    - Better Soft Skills Ratings are linked to improved placement outcomes.""")

st.markdown("----")
st.subheader("Heatmap")
image=Image.open("images/heatmap.png")
st.image(image,width='content')
st.markdown("""
       INSIGHT :
    - Most features show a positive correlation with Placement Status, indicating that better academic performance, skills, and practical experience are associated with higher placement chances.
    - Aptitude Test Score and HSC Marks exhibit relatively stronger relationships with Placement Status compared to other features.
    - Projects and Extracurricular Activities also demonstrate a notable positive association with placement outcomes.
    - CGPA, SSC Marks, and Soft Skills Rating contribute positively toward placement prediction.
    - No pair of features shows extremely high correlation, suggesting the absence of severe multicollinearity in the dataset..""")








