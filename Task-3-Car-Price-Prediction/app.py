import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Page settings
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)


# Title
st.title("🚗 Car Price Prediction")

st.write(
    "Enter the car details below to predict "
    "the selling price using Machine Learning."
)

st.markdown("---")


# Load dataset
df = pd.read_csv(
    "Task-3-Car-Price-Prediction/car_data.csv"
)


# Remove duplicates
df = df.drop_duplicates().reset_index(drop=True)


# Clean categorical columns
cat_cols = [
    "Fuel_Type",
    "Seller_Type",
    "Transmission",
    "Car_Name"
]

for col in cat_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()


# Feature engineering
CURRENT_YEAR = 2020

df["Car_Age"] = CURRENT_YEAR - df["Year"]

df["Brand"] = df["Car_Name"].str.split().str[0]


# Select required features
df_model = df[
    [
        "Selling_Price",
        "Present_Price",
        "Kms_Driven",
        "Fuel_Type",
        "Seller_Type",
        "Transmission",
        "Owner",
        "Car_Age",
        "Brand"
    ]
]


# One-hot encoding
df_encoded = pd.get_dummies(
    df_model,
    columns=[
        "Fuel_Type",
        "Seller_Type",
        "Transmission",
        "Brand"
    ],
    drop_first=True
)


# Features and target
X = df_encoded.drop(
    columns=["Selling_Price"]
)

y = df_encoded["Selling_Price"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Linear Regression
linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)


# Random Forest
rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)


# Predictions
linear_predictions = linear_model.predict(
    X_test
)

rf_predictions = rf_model.predict(
    X_test
)


# Linear Regression metrics
linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = mean_squared_error(
    y_test,
    linear_predictions
) ** 0.5

linear_r2 = r2_score(
    y_test,
    linear_predictions
)


# Random Forest metrics
rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = mean_squared_error(
    y_test,
    rf_predictions
) ** 0.5

rf_r2 = r2_score(
    y_test,
    rf_predictions
)


# Input section
st.subheader("🚘 Enter Car Details")


# Present Price
present_price = st.number_input(
    "Present Price (₹ lakh)",
    min_value=0.1,
    max_value=100.0,
    value=8.0,
    step=0.1
)


# Kilometers Driven
kms_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    max_value=500000,
    value=30000,
    step=1000
)


# Car Age
car_age = st.number_input(
    "Car Age (years)",
    min_value=0,
    max_value=50,
    value=5,
    step=1
)


# Previous Owners
owner = st.number_input(
    "Previous Owners",
    min_value=0,
    max_value=5,
    value=0,
    step=1
)


# Fuel Type
fuel_type = st.text_input(
    "Fuel Type",
    value="petrol"
).strip().lower()


# Seller Type
seller_type = st.text_input(
    "Seller Type",
    value="dealer"
).strip().lower()


# Transmission
transmission = st.text_input(
    "Transmission",
    value="manual"
).strip().lower()


# Brand
brand = st.text_input(
    "Brand / Model Family",
    value="ritz"
).strip().lower()


st.caption(
    "Enter categorical values such as petrol/diesel/cng, "
    "dealer/individual, and manual/automatic."
)


# Prediction button
if st.button("🔮 Predict Car Price"):

    # Create input dataframe
    input_data = pd.DataFrame(
        {
            "Present_Price": [present_price],
            "Kms_Driven": [kms_driven],
            "Fuel_Type": [fuel_type],
            "Seller_Type": [seller_type],
            "Transmission": [transmission],
            "Owner": [owner],
            "Car_Age": [car_age],
            "Brand": [brand]
        }
    )


    # One-hot encode input
    input_encoded = pd.get_dummies(
        input_data,
        columns=[
            "Fuel_Type",
            "Seller_Type",
            "Transmission",
            "Brand"
        ],
        drop_first=True
    )


    # Match training columns
    input_encoded = input_encoded.reindex(
        columns=X.columns,
        fill_value=0
    )


    # Predict using Random Forest
    prediction = rf_model.predict(
        input_encoded
    )[0]


    # Display prediction
    st.markdown("---")

    st.subheader("🚗 Prediction Result")

    st.success(
        f"Predicted Selling Price: "
        f"**₹{prediction:.2f} lakh**"
    )


# Model performance
st.markdown("---")

st.subheader("🤖 Model Performance")


col1, col2, col3 = st.columns(3)


with col1:
    st.metric(
        "RF MAE",
        f"{rf_mae:.3f}"
    )


with col2:
    st.metric(
        "RF RMSE",
        f"{rf_rmse:.3f}"
    )


with col3:
    st.metric(
        "RF R²",
        f"{rf_r2:.3f}"
    )


# Model comparison
st.subheader("📊 Model Comparison")


comparison = pd.DataFrame(
    {
        "Model": [
            "Linear Regression",
            "Random Forest Regressor"
        ],
        "MAE": [
            linear_mae,
            rf_mae
        ],
        "RMSE": [
            linear_rmse,
            rf_rmse
        ],
        "R²": [
            linear_r2,
            rf_r2
        ]
    }
)


comparison["MAE"] = comparison["MAE"].round(3)

comparison["RMSE"] = comparison["RMSE"].round(3)

comparison["R²"] = comparison["R²"].round(3)


st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)


# Dataset information
st.markdown("---")

st.subheader("📚 About the Dataset")

st.write(
    "The dataset contains used-car listings with "
    "information about price, mileage, fuel type, "
    "seller type, transmission, and previous owners."
)

st.write(
    "The application uses an 80/20 train-test split "
    "and compares Linear Regression with Random Forest "
    "Regression."
)


# Footer
st.markdown("---")

st.caption(
    "OIBSIP Data Science Internship — Task 3"
)
