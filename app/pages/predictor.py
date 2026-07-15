import streamlit as st
import requests

st.set_page_config(page_title="Student Placement Prediction System",
                   page_icon="🎓",
                   layout="wide")

st.title("🎓Student Placement Prediction System")
st.markdown(
    """
    Welcome! This application predicts whether a student is likely
    to be placed based on academic performance, technical skills,
    internships, and extracurricular activities.
    Fill in the student details below to predict placement status.
    """
)

st.header("Student Details")
col1,col2=st.columns(2)
with col1:
    cgpa=st.number_input("CGPA",min_value=0.0,max_value=10.0,value=7.5,step=0.1)
    internships=st.number_input("Internships",min_value=0,step=1)
    projects_count=st.number_input("Project Count",min_value=0,step=1)
    certifications = st.number_input("Workshops/Certifications",min_value=0,step=1)
    aptitude_score=st.slider("Aptitude Test Score",0,100,50)
    
with col2:
    soft_skills = st.number_input("Soft Skills Rating",min_value=3.0,max_value=5.0,value=4.0,step=0.1)
    ssc_marks = st.number_input("Secondry School Marks",min_value=0,step=1,value=70)
    hsc_marks = st.number_input("High School Marks",min_value=0,step=1,value=75)
    placement_training = st.selectbox("Placement Training Attended?",["No", "Yes"])
    extracurricular = st.selectbox("Participated in Extracurricular Activities?",["No", "Yes"])

predict_button=st.button("🚀Predict Placement",use_container_width=True)
if predict_button:
    st.write("Prediction Started....")
    placement_training = 1 if placement_training == "Yes" else 0
    extracurricular = 1 if extracurricular == "Yes" else 0

    payload={
    "CGPA": cgpa,
    "Internships": internships,
    "Projects": projects_count,
    "Workshops_Certifications": certifications,
    "AptitudeTestScore": aptitude_score,
    "SoftSkillsRating": soft_skills,
    "ExtracurricularActivities": extracurricular,
    "PlacementTraining" : placement_training,
    "SSC_Marks": ssc_marks,
    "HSC_Marks": hsc_marks }

    response=requests.post("http://127.0.0.1:8000/predict",json=payload,timeout=5)   
    result=response.json()
    
    st.write(response.status_code)

    prediction = result["Prediction"]
    placed_prob = result["Placed Probability (%)"]
    not_placed_prob = result["Not Placed Probability (%)"]

    if prediction == 1:
        st.success("🎉 Student is likely to be Placed")
    else:
        st.error("⚠️ Student is unlikely to be Placed")
    st.subheader("Prediction Probabilities")
    col1, col2 = st.columns(2)

    with col1:
         st.metric("🎉 Probability of Placement",f"{placed_prob:.2f}%")

    with col2:
         st.metric("⚠️ Probability of not being Placed",f"{not_placed_prob:.2f}%")
 