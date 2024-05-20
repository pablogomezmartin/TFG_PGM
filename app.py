import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.evaluate import create_counterfactual
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# Ponemos el título de la página
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

#Tomamos el directorio de trabajo del archivo principal
working_dir = os.path.dirname(os.path.abspath(__file__))

diabetes_dataset = pd.read_csv('./dataset/diabetes.csv') 
X_dataset = diabetes_dataset.drop('Outcome', axis=1).values

# Cargar el modelo, pero el estimador que se guardó en el pipeline
diabetes_model = pickle.load(open(f'{working_dir}/model.pkl', 'rb'))
best_estimator = diabetes_model.best_estimator_

# Imprimir todos los pasos en el pipeline
#print(best_estimator.named_steps)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Barra lateral , para establecer una interfaz de usuario
with st.sidebar:
    selected = option_menu('SISTEMA DE PREDICCIÓN',
                           ['PREDICIÓN DE DIABETES'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart'],
                           default_index=0)


# Variables para controlar los valores de lambda , ponermos los valores mínimos, máximos y por defecto (a 0 para que sea mínimo)
lambda_min = 0.0
lambda_max = 1.0
default_lambda = 0.0


# Sección de predicción de diabetes pagina principal 
if selected == 'PREDICIÓN DE DIABETES':
    # Ti tulo de la página
    st.title('PREDICCIÓN DE DIABETES')
    
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
    
    with col3:
        lambda_value = st.slider('Lambda', min_value=lambda_min, max_value=lambda_max, step=0.01, value=default_lambda)

    diab_diagnosis = ''
    
    diabetes_test_executed = False

    # Botón de predicción para ejecutar los datos introducidos por usuario
    if st.button('Diabetes Test'):
        diabetes_test_executed = True
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
        if diabetes_test_executed:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.title('CONTRAEJEMPLO')
                # Crear un contrafactual con la libreía mlxtend
                counterfactual = create_counterfactual(model=diabetes_model, X_dataset=X_dataset, x_reference=user_data_normalized.values, y_desired=y_desired,lammbda=lambda_value)

                # Desnormalizar los datos del contrafactual
                counterfactual = pd.DataFrame(scaler.inverse_transform([counterfactual]), columns=user_data.columns)
                # Mostrar el contrafactual en Streamlit, generación del gráfico con plotly
                fig = go.Figure(data=[
                    go.Bar(name='Característica Contraejemplo', y=user_data.columns, x=counterfactual.values[0], orientation='h', 
                           marker_color=['lightblue' if val < 0 else 'salmon' for val in counterfactual.values[0]])
                ])
                fig.update_layout(title_text='Contraejemplo')
                st.plotly_chart(fig)

                # Gráfico de barras con Matplotlib , queda demasiado grande y feo
                #fig, ax = plt.subplots(figsize=(8, 8))
                #colors = ['lightblue' if val < 0 else 'salmon' for val in counterfactual.values[0]]
                #ax.barh(user_data.columns, counterfactual.values[0], color=colors)
                #ax.set_title('Contraejemplo')
                #st.pyplot(fig)

            with col2:
                #Mostrar la predicción con los valores del contrafactual
                diab_prediction_cf = diabetes_model.predict_proba(counterfactual)
                #st.success(f'Etiqueta predicha: {diab_prediction_cf[0].argmax()}')

            with col3:
                # Analisis de sensibilidad, título
                st.title('ANÁLISIS DE SENSIBILIDAD')

                feature_influence = {}

                # Para cada característica en el conjunto de datos
                for feature in user_data.columns:
                    # Creación de una copia de los datos del usuario, mismo tmaño
                    user_data_copy = user_data_normalized.copy()
                    
                    # Copiamos los valores del contrafactual
                    user_data_copy[feature] = counterfactual[feature]
                    
                    # Hacer una predicción con el modelo utilizando los datos modificados
                    new_prediction = diabetes_model.predict(user_data_copy)
                    
                    # Diferencia , dato importante para ver como aprende
                    feature_influence[feature] = abs(diab_prediction[0] - new_prediction[0])

                # Ordenar por la mayor diferencia en la predicción
                feature_influence_sorted = sorted(feature_influence.items(), key=lambda x: x[1], reverse=True)

                # Convertir el diccionario en un DataFrame de Pandas para mostrarlo de manera más visual
                feature_influence_df = pd.DataFrame(feature_influence_sorted, columns=['Feature', 'Influence'])
                
                # Mostrar el dataframe
                st.dataframe(feature_influence_df)