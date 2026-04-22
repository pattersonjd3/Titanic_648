import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open("survival_model.pkl", "rb"))

st.title('Titanic Survival Prediction')
st.write('Enter passenger details to predict survival chance.')

# Create input fields for user
pclass = st.selectbox('PClass (Passenger Class)', [1, 2, 3])
age = st.slider('Age', 0.42, 80.0, 30.0)
sex = st.selectbox('Sex', ['male', 'female'])
fare = st.number_input('Fare', min_value=0.0, max_value=512.3292, value=30.0)

# When the user clicks the 'Predict' button
if st.button('Predict Survival'):
    # Create a DataFrame from the input data
    input_data = pd.DataFrame([[pclass, age, sex, fare]], 
                              columns=['Pclass', 'Age', 'Sex', 'Fare'])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[:, 1][0]
    
    # Display result
    st.subheader('Prediction Result:')
    if prediction == 1:
        st.success(f'The passenger is predicted to survive with a probability of {prediction_proba:.2f}.')
    else:
        st.error(f'The passenger is predicted not to survive with a probability of {1 - prediction_proba:.2f}.')
