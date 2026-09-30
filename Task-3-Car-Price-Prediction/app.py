import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Car Price Prediction")
st.write(
    "Predict the selling price of a used car based on "
    "vehicle-related features."
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

# Select features
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

X = df_encoded.drop(columns=["Selling_Price"])
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
linear_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)
rf_model.fit(X_train, y_train)

# Predictions
linear_predictions = linear_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

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


# Sidebar
st.sidebar.header("🚘 Car Details")

present_price = st.sidebar.number_input(
    "Present Price (₹ lakh)",
    min_value=0.1,
    max_value=100.0,
    value=8.0,
    step=0.1
)

kms_driven = st.sidebar.number_input(
    "Kilometers Driven",
    min_value=0,
    max_value=500000,
    value=30000,
    step=1000
)

car_age = st.sidebar.slider(
    "Car Age (years)",
    min_value=0,
    max_value=20,
    value=5
)

owner = st.sidebar.selectbox(
    "Previous Owners",
    [0, 1, 2, 3]
)

fuel_type = st.sidebar.selectbox(
    "Fuel Type",
    ["petrol", "diesel", "cng"]
)

seller_type = st.sidebar.selectbox(
    "Seller Type",
    ["dealer", "individual"]
)

transmission = st.sidebar.selectbox(
    "Transmission",
    ["manual", "automatic"]
)

brands = sorted(df["Brand"].unique())

brand = st.sidebar.selectbox(
    "Brand / Model Family",
    brands
)


# Prediction
if st.button("🔮 Predict Car Price"):

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

    prediction = rf_model.predict(
        input_encoded
    )[0]

    st.success(
        f"🚗 Predicted Selling Price: "
        f"**₹{prediction:.2f} lakh**"
    )


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


st.markdown("---")

st.subheader("📚 Dataset Information")

st.write(
    "The dataset contains used-car listings with "
    "information about price, mileage, fuel type, "
    "seller type, transmission, and previous owners."
)

st.write(
    "The model uses an 80/20 train-test split and "
    "compares Linear Regression with Random Forest "
    "Regression."
)

st.caption(
    "OIBSIP Data Science Internship — Task 3"
)
