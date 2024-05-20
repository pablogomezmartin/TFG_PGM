import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from mlxtend.evaluate import create_counterfactual
import plotly.graph_objects as go

# Ponemos el título de la página
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Tomamos el directorio de trabajo del archivo principal
working_dir = os.path.dirname(os.path.abspath(__file__))

# Cargar el dataset de diabetes
diabetes_dataset = pd.read_csv('./dataset/diabetes.csv')
X_dataset = diabetes_dataset.drop('Outcome', axis=1).values

# Cargar el modelo
diabetes_model = pickle.load(open(f'{working_dir}/model.pkl', 'rb'))
best_estimator = diabetes_model.best_estimator_

# Cargar el scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Barra lateral para establecer una interfaz de usuario
with st.sidebar:
    selected = option_menu('SISTEMA DE PREDICCIÓN',
                           ['PREDICIÓN DE DIABETES'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart'],
                           default_index=0)

# Sección de predicción de diabetes en la página principal
if selected == 'PREDICIÓN DE DIABETES':
    # Título de la página
    st.title('PREDICIÓN DE DIABETES')

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.slider('Pregnancies', min_value=0, max_value=30)

    with col2:
        Glucose = st.slider('Glucose', min_value=0, max_value=200)

    with col3:
        BloodPressure = st.slider('Blood Pressure', min_value=0, max_value=150)

    with col1:
        SkinThickness = st.slider('Skin Thickness', min_value=0, max_value=100)

    with col2:
        Insulin = st.slider('Insulin', min_value=0, max_value=900)

    with col3:
        BMI = st.slider('BMI', min_value=0.0, max_value=90.0, step=0.01)

    with col1:
        DiabetesPedigreeFunction = st.slider('DiabetesPedigreeFunction', min_value=0.0, max_value=3.0, step=0.01)

    with col2:
        Age = st.slider('Age', min_value=0, max_value=120)

    diab_diagnosis = ''

    # Botón de predicción para ejecutar los datos introducidos por el usuario
    if st.button('Diabetes Test'):
        user_data = pd.DataFrame({
            'Pregnancies': [Pregnancies],
            'Glucose': [Glucose],
            'BloodPressure': [BloodPressure],
            'SkinThickness': [SkinThickness],
            'Insulin': [Insulin],
            'BMI': [BMI],
            'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
            'Age': [Age]
        })

        # Normalizar los datos del usuario necesario para la entrada de datos del modelo
        user_data_normalized = pd.DataFrame(scaler.transform(user_data), columns=user_data.columns)

        diab_prediction = diabetes_model.predict(user_data_normalized)

        if diab_prediction[0] == 1:
            diab_diagnosis = 'La persona SI tiende a tener diabetes (1)'
            y_desired = 0
        else:
            diab_diagnosis = 'La persona NO tiende a tener diabetes (0)'
            y_desired = 1

        st.success(diab_diagnosis)

        # Automatización de lambda, escofiendo el valor óptimo desde 0 a 1 de pasos de 11
        def evaluate_lambda(lam):
            counterfactual = create_counterfactual(
                model=diabetes_model, X_dataset=X_dataset,
                x_reference=user_data_normalized.values,
                y_desired=y_desired, lammbda=lam
            )
            diff = np.sum(np.abs(user_data_normalized.values - counterfactual))
            return diff

        lambda_values = np.linspace(0, 1, 20)  # Probar 11 valores entre 0 y 1
        best_lambda = min(lambda_values, key=evaluate_lambda)

        st.write(f'Lambda óptimo: {best_lambda}')

        counterfactual = create_counterfactual(
            model=diabetes_model, X_dataset=X_dataset,
            x_reference=user_data_normalized.values,
            y_desired=y_desired, lammbda=best_lambda
        )

        counterfactual = pd.DataFrame(scaler.inverse_transform([counterfactual]), columns=user_data.columns)

        # Mostrar el contrafactual en Streamlit, generación del gráfico con plotly
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure(data=[
                go.Bar(name='Característica Contraejemplo', y=user_data.columns, x=counterfactual.values[0], orientation='h',
                       marker_color=['lightblue' if val < 0 else 'salmon' for val in counterfactual.values[0]])
            ])
            fig.update_layout(title_text='Contraejemplo')
            st.plotly_chart(fig)

        with col2:
            st.title('ANÁLISIS DE SENSIBILIDAD')

            feature_influence = {}

            for feature in user_data.columns:
                user_data_copy = user_data_normalized.copy()
                user_data_copy[feature] = counterfactual[feature]
                new_prediction = diabetes_model.predict(user_data_copy)
                feature_influence[feature] = abs(diab_prediction[0] - new_prediction[0])

            feature_influence_sorted = sorted(feature_influence.items(), key=lambda x: x[1], reverse=True)
            feature_influence_df = pd.DataFrame(feature_influence_sorted, columns=['Feature', 'Influence'])
            st.dataframe(feature_influence_df)
