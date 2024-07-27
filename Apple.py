import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.title('Apple Quality Prediction')

def user_input_features():
    try:
        Size = st.slider('Apple Size', -10.0, 10.0, 0.0)
        Weight = st.slider('Apple Weight', -10.0, 10.0, 0.0)
        Sweetness = st.slider('Apple Sweetness', -10.0, 10.0, 0.0)
        Crunchiness = st.slider('Apple Crunchiness', -10.0, 10.0, 0.0)
        Juiciness = st.slider('Apple Juiciness', -10.0, 10.0, 0.0)
        Ripeness = st.slider('Apple Ripness', -10.0, 10.0, 0.0)
        Acidity = st.slider('Apple Acidity', -10.0, 10.0, 0.0)
    except ValueError:
        st.error('Please enter the numeric value.')

    data = {
        'Size' : Size,
        'Weight' : Weight,
        'Sweetness' : Sweetness,
        'Crunchiness' : Crunchiness,
        'Juiciness' : Juiciness,
        'Ripeness' : Ripeness,
        'Acidity' : Acidity
    }

    features = pd.DataFrame(data, index = [0])
    return features

input_df = user_input_features()

if input_df is not None:
    with open(r'E:\BOOKS\Bs Eco\Applied Machine Learning\Machine Learning\Deployment_models\Apple Quality\std_scaler.pkl', 'rb') as f:
        std_scaler = pickle.load(f)
    with open(r'E:\BOOKS\Bs Eco\Applied Machine Learning\Machine Learning\Deployment_models\Apple Quality\knn_model.pkl', 'rb') as f:
        knn = pickle.load(f)
    with open(r'E:\BOOKS\Bs Eco\Applied Machine Learning\Machine Learning\Deployment_models\Apple Quality\dt_model.pkl', 'rb') as f:
        dt = pickle.load(f)
    with open(r'E:\BOOKS\Bs Eco\Applied Machine Learning\Machine Learning\Deployment_models\Apple Quality\rf_model.pkl', 'rb') as f:
        rf = pickle.load(f)

    input_scaled = std_scaler.transform(input_df)

    model_choice = st.selectbox('Choose the model',
                                ['KNN', 'Decision Tree', 'Random Forest'])

    if model_choice == 'KNN':
        model = knn
    elif model_choice == 'Decision Tree':
        model = dt
    else:
        model = rf

    if st.button('Predict'):
        try:
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)
        except:
            st.error('Error occured during prediction..!!')

        st.subheader('Prediction')

        if prediction == 0:
            st.write('Apple Quality: Good')
        else:
            st.write('Apple Quality: Not Good')

        st.subheader('Probability')
        if prediction == 0:
            st.write(f'Probability: {probability[0][0]: 0.3f}')
        else:
            st.write(f'Probability: {probability[0][1]: 0.3f}')




