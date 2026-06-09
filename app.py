import streamlit as st
import pandas as pd
import joblib

model = joblib.load("house_price_xgboost_model.pkl")

st.title("🏠 House Price Prediction using XGBoost")

st.markdown("""
This application uses an XGBoost Regressor model trained on the Ames Housing Dataset to estimate house sale prices based on selected house characteristics.
""")

st.caption("Model: XGBoost Regressor | Dataset: Ames Housing Dataset")

overall_qual = st.slider("Overall Quality (1 = Poor, 10 = Excellent)", 1, 10, 5)
gr_liv_area = st.number_input("Above Ground Living Area / GrLivArea", min_value=0, value=1500)
garage_cars = st.number_input("Garage Cars", min_value=0, value=2)
garage_area = st.number_input("Garage Area", min_value=0, value=500)
total_bsmt_sf = st.number_input("Total Basement Area / TotalBsmtSF", min_value=0, value=1000)
first_flr_sf = st.number_input("First Floor Area / 1stFlrSF", min_value=0, value=1200)
full_bath = st.number_input("Full Bathrooms", min_value=0, value=2)
year_built = st.number_input("Year Built", min_value=1800, max_value=2026, value=2000)

if st.button("Predict House Price"):
    input_data = pd.DataFrame({
        'OverallQual': [overall_qual],
        'GrLivArea': [gr_liv_area],
        'GarageCars': [garage_cars],
        'GarageArea': [garage_area],
        'TotalBsmtSF': [total_bsmt_sf],
        '1stFlrSF': [first_flr_sf],
        'FullBath': [full_bath],
        'YearBuilt': [year_built]
    })

    prediction = model.predict(input_data)

    st.success(f"🏡 Estimated House Price: ${prediction[0]:,.2f}")