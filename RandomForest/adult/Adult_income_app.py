import streamlit as st
import pandas as pd
from Adult_income_Randomforest import *
import pickle
def app():
    with open('my_model.pkl', 'rb') as file:
        model = pickle.load(file)
    df=data_loading()
    df.columns = ['age','job_type','person_weight','education_level','education_years',
              'marital_status','job_role','family_role','ethnicity','gender','capital_gain',
              'capital_loss','weekly_hours','country_of_origin','income']

   

    
    st.title("Adult Income Prediction")
    age = st.number_input("Enter your age:", min_value=0, max_value=100, step=1)
    job_type = st.selectbox("Select your job type:", df['job_type'].unique())
    person_weight = st.number_input("Enter your weight:", min_value=0, max_value=200, step=1)
    education_level = st.selectbox("Select your education level:", df['education_level'].unique())
    education_years = st.number_input("Enter your education years:", min_value=0, max_value=20, step=1)
    marital_status = st.selectbox("Select your marital status:", df['marital_status'].unique())
    job_role = st.selectbox("Select your job role:", df['job_role'].unique())
    family_role = st.selectbox("Select your family role:", df['family_role'].unique())
    ethnicity = st.selectbox("Select your ethnicity:", df['ethnicity'].unique())
    gender = st.selectbox("Select your gender:", df['gender'].unique())
    capital_gain = st.number_input("Enter your capital gain:", min_value=0, max_value=100000, step=1)
    capital_loss = st.number_input("Enter your capital loss:", min_value=0, max_value=100000, step=1)
    weekly_hours = st.number_input("Enter your weekly hours:", min_value=0, max_value=100, step=1)
    country_of_origin = st.selectbox("Select your country of origin:", df['country_of_origin'].unique())

    
    if st.button("Predict"):
       
            

    # Load the already-fitted scaler
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        input_data = pd.DataFrame([[age,job_type,person_weight,education_level,education_years,
                                    marital_status,job_role,family_role,ethnicity,gender,capital_gain,
                                    capital_loss,weekly_hours,country_of_origin]],
                    columns=['age','job_type','person_weight','education_level','education_years',
                                'marital_status','job_role','family_role','ethnicity','gender','capital_gain',
                                    'capital_loss','weekly_hours','country_of_origin'])
       
        df=adding_new_features(input_data)
        # input_data = adding_new_features(input_data)
        df=data_encoding(input_data)
        data=scaler.transform(df)
        prediction = model.predict(data)

        class_names = ['<=50K', '>50K']
        predicted_class_name = class_names[int(prediction[0])]
        st.title(f"The Income of the person is: {predicted_class_name}")

    
if __name__ == "__main__":
    app()