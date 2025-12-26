import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


page=st.sidebar.selectbox('Page Navigation', ["Purchase Prediction","Price Prediction","Clustering","About"])
st.sidebar.markdown("""-----""")
st.sidebar.write('Created by [Mohamed Afrith](https://www.linkedin.com/in/mohamed-afrit-s/)')
if page == 'Purchase Prediction':
    # Set the title of the app
    st.title("Predict Customer Purchase from Browsing Behavior")

    # Load your trained model from the local pickle file
    try:
        with open('/Users/mohamedafrith/Desktop/mini_project_5/mlruns/385524123148715307/ec9d402b97984ab1899ba9ed7f86b618/artifacts/DecisionTreeClassifier/model.pkl', 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        st.error("The model file 'model-2.pkl' was not found in the specified directory. Please check the file path.")
        st.stop()

    # Create the user input section
    st.header("Enter the details to predict whether a customer will complete a purchase or not:")

    year= st.number_input("Year", min_value=2008, value=2008)

    month = st.number_input("Month", min_value=1, max_value=12, value=1)

    day = st.number_input("Day", min_value=1, max_value=31, value=1)

    order=st.number_input("Sequence of clicks during one session",value=1)

    country=st.selectbox("Choose Country",["Austria","Belgium","Croatia","Cyprus","Czech","Republic","Denmark","Estonia","Finland","France","Germany","Greece","Hungary","Iceland","Ireland","Italy","Latvia","Lithuania","Luxembourg","Netherlands","Norway","Poland","Portugal","Romania","Russia","San Marino","Slovakia","Slovenia","Spain","Sweden","Switzerland","Ukraine","United Kingdom","Others"])

    session_id=st.number_input("Session ID",value=0)

    page1_main_category=st.selectbox("MAIN CATEGORY",['Trousers','Skirts','Blouses','Sale'])

    page2_clothing_model=st.number_input("Code for each product",value=0)

    colour=st.selectbox("Colour",['beige','black','blue','brown','burgundy','gray','green','navy', 'blue','of many colors','olive','pink','red','violet','white'])

    location=st.selectbox("photo location on the page",['top left','top in the middle','top right','bottom left','bottom in the middle','bottom right'])
    
    model_photography=st.selectbox("MODEL PHOTOGRAPHY",['en face','profile'])

    price=st.number_input("PRICE",value=1)

    page=st.selectbox("Page number within the e-store website",[1,2,3,4,5])




    # When the predict button is clicked
    if st.button("Predict customer will complete a purchase or not"):
        # Map the company to its corresponding numeric value
        country_mapping = {
                            "Austria": 0,
                            "Belgium": 1,
                            "Croatia": 2,
                            "Cyprus": 3,
                            "Czech Republic": 4,
                            "Denmark": 5,
                            "Estonia": 6,
                            "Finland": 7,
                            "France": 8,
                            "Germany": 9,
                            "Greece": 10,
                            "Hungary": 11,
                            "Iceland": 12,
                            "Ireland": 13,
                            "Italy": 14,
                            "Latvia": 15,
                            "Lithuania": 16,
                            "Luxembourg": 17,
                            "Netherlands": 18,
                            "Norway": 19,
                            "Poland": 20,
                            "Portugal": 21,
                            "Romania": 22,
                            "Russia": 23,
                            "San Marino": 24,
                            "Slovakia": 25,
                            "Slovenia": 26,
                            "Spain": 27,
                            "Sweden": 28,
                            "Switzerland": 29,
                            "Ukraine": 30,
                            "United Kingdom": 31,
                            "Others": 32}


        country_mapping= country_mapping.get(country, -1)
        category_mapping = {
                                "Trousers": 0,
                                "Skirts": 1,
                                "Blouses": 2,
                                "Sale": 3}
        category_mapping= category_mapping.get(page1_main_category, -1)
        color_mapping = {
                            "beige": 0,
                            "black": 1,
                            "blue": 2,
                            "brown": 3,
                            "burgundy": 4,
                            "gray": 5,
                            "green": 6,
                            "navy blue": 7,
                            "of many colors": 8,
                            "olive": 9,
                            "pink": 10,
                            "red": 11,
                            "violet": 12,
                            "white": 13}
        color_mapping= color_mapping.get(colour, -1)
        position_mapping = {
                            "top left": 0,
                            "top in the middle": 1,
                            "top right": 2,
                            "bottom left": 3,
                            "bottom in the middle": 4,
                            "bottom right": 5}
        position_mapping= position_mapping.get(location, -1)

        face_orientation_mapping = {
                            "en face": 0,
                            "profile": 1}
        face_orientation_mapping= face_orientation_mapping.get(model_photography, -1)






    


        # Prepare the features array. Ensure the order matches the one used during model training.
        # In this example, the order is: [company, open_price, day, year, month]
        features = np.array([[year,	month,	day,	order,	country_mapping,	session_id,	category_mapping,	page2_clothing_model,	color_mapping,	position_mapping,	face_orientation_mapping,	price,	page]])
        
        try:
            # Make a prediction using the model
            prediction = model.predict(features)
            # Convert the predicted value to a float
            purchase = int(prediction)
            if purchase==1:
                purchase='Yes'
            else:
                purchase='No'

            st.success(f"{purchase}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
if page == "Price Prediction":
    # Set the title of the app
    st.title("Price Prediction")

    # Load your trained model from the local pickle file
    try:
        with open('/Users/mohamedafrith/Desktop/mini_project_5/mlruns/385524123148715307/2220ed1c56d54a1f9369efadf2da6026/artifacts/DecisionTreeRegressor/model.pkl', 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        st.error("The model file 'model-2.pkl' was not found in the specified directory. Please check the file path.")
        st.stop()

    # Create the user input section
    st.header("Enter the details to predict price")

    year= st.number_input("Year", min_value=2008, value=2008)

    month = st.number_input("Month", min_value=1, max_value=12, value=1)

    day = st.number_input("Day", min_value=1, max_value=31, value=1)

    order=st.number_input("Sequence of clicks during one session",value=1)

    country=st.selectbox("Choose Country",["Austria","Belgium","Croatia","Cyprus","Czech","Republic","Denmark","Estonia","Finland","France","Germany","Greece","Hungary","Iceland","Ireland","Italy","Latvia","Lithuania","Luxembourg","Netherlands","Norway","Poland","Portugal","Romania","Russia","San Marino","Slovakia","Slovenia","Spain","Sweden","Switzerland","Ukraine","United Kingdom","Others"])

    session_id=st.number_input("Session ID",value=0)

    page1_main_category=st.selectbox("MAIN CATEGORY",['Trousers','Skirts','Blouses','Sale'])

    page2_clothing_model=st.number_input("Code for each product",value=0)

    colour=st.selectbox("Colour",['beige','black','blue','brown','burgundy','gray','green','navy', 'blue','of many colors','olive','pink','red','violet','white'])

    location=st.selectbox("photo location on the page",['top left','top in the middle','top right','bottom left','bottom in the middle','bottom right'])
    
    model_photography=st.selectbox("MODEL PHOTOGRAPHY",['en face','profile'])

    price_2=st.selectbox("Purchased",[1,2])

    page=st.selectbox("Page number within the e-store website",[1,2,3,4,5])




    # When the predict button is clicked
    if st.button("Predict Price"):
        # Map the company to its corresponding numeric value
        country_mapping = {
                            "Austria": 0,
                            "Belgium": 1,
                            "Croatia": 2,
                            "Cyprus": 3,
                            "Czech Republic": 4,
                            "Denmark": 5,
                            "Estonia": 6,
                            "Finland": 7,
                            "France": 8,
                            "Germany": 9,
                            "Greece": 10,
                            "Hungary": 11,
                            "Iceland": 12,
                            "Ireland": 13,
                            "Italy": 14,
                            "Latvia": 15,
                            "Lithuania": 16,
                            "Luxembourg": 17,
                            "Netherlands": 18,
                            "Norway": 19,
                            "Poland": 20,
                            "Portugal": 21,
                            "Romania": 22,
                            "Russia": 23,
                            "San Marino": 24,
                            "Slovakia": 25,
                            "Slovenia": 26,
                            "Spain": 27,
                            "Sweden": 28,
                            "Switzerland": 29,
                            "Ukraine": 30,
                            "United Kingdom": 31,
                            "Others": 32}


        country_mapping= country_mapping.get(country, -1)
        category_mapping = {
                                "Trousers": 0,
                                "Skirts": 1,
                                "Blouses": 2,
                                "Sale": 3}
        category_mapping= category_mapping.get(page1_main_category, -1)
        color_mapping = {
                            "beige": 0,
                            "black": 1,
                            "blue": 2,
                            "brown": 3,
                            "burgundy": 4,
                            "gray": 5,
                            "green": 6,
                            "navy blue": 7,
                            "of many colors": 8,
                            "olive": 9,
                            "pink": 10,
                            "red": 11,
                            "violet": 12,
                            "white": 13}
        color_mapping= color_mapping.get(colour, -1)
        position_mapping = {
                            "top left": 0,
                            "top in the middle": 1,
                            "top right": 2,
                            "bottom left": 3,
                            "bottom in the middle": 4,
                            "bottom right": 5}
        position_mapping= position_mapping.get(location, -1)

        face_orientation_mapping = {
                            "en face": 0,
                            "profile": 1}
        face_orientation_mapping= face_orientation_mapping.get(model_photography, -1)






    


        # Prepare the features array. Ensure the order matches the one used during model training.
        # In this example, the order is: [company, open_price, day, year, month]
        features = np.array([[year,	month,	day,	order,	country_mapping,	session_id,	category_mapping,	page2_clothing_model,	color_mapping,	position_mapping,	face_orientation_mapping,	price_2,	page]])
        
        try:
            # Make a prediction using the model
            prediction = model.predict(features)
            # Convert the predicted value to a float
            price = int(prediction)
            
                
            st.success(f"${price}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
if page == "Clustering":
    # Set the title of the app
    st.title("Clustering Visualization")
    st.image('/Users/mohamedafrith/Desktop/mini_project_5/output_2.png')


if page=='About':

    st.title('Customer Conversion Analysis for Online Shopping Using Clickstream Data')#project title
    st.header('E-commerce and Retail Analytics')
    st.header("Introduction:", divider='gray')#header
    st.markdown("This project develops a **Streamlit web application** that analyzes **clickstream data** to predict purchases, estimate revenue, and segment customers. By leveraging **machine learning models**, businesses can enhance customer engagement, optimize marketing, and boost sales.")
    st.header('Approach:',divider="gray")
    st.markdown("This project follows a structured approach, from **data preprocessing and EDA** to **feature engineering, model building, evaluation, and deployment via a Streamlit app**, enabling real-time predictions, revenue estimation, and customer segmentation.")
    
    
    st.header("Skills Takeaway:",divider='gray')
    




    st.markdown("This project encompasses **data preprocessing, EDA, and feature engineering**, followed by **supervised and unsupervised learning techniques** for classification, regression, and clustering. It includes **model evaluation, hyperparameter tuning, and pipeline development**, culminating in a **Streamlit-based interactive deployment** for real-time insights and predictions.")#text content
    
    st.header("Result:" ,divider='gray')
    st.markdown("The project successfully **predicts customer purchase behavior, estimates potential revenue, and segments customers based on browsing patterns** using machine learning models. The **Streamlit application provides real-time predictions, interactive visualizations, and personalized insights**, empowering businesses to enhance customer engagement, optimize marketing strategies, and drive sales.")
