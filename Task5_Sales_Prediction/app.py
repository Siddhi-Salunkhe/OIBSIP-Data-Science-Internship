import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Sales Prediction",
    page_icon="📈",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📈 Sales Prediction Using Python")

st.write(
    "Predict product sales based on advertising "
    "spending across TV, Radio, and Newspaper."
)

st.markdown("---")


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv("Task5_Sales_Prediction/Advertising.csv", index_col=0)

# --------------------------------------------------
# Features and Target
# --------------------------------------------------

X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]


# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# Linear Regression
# --------------------------------------------------

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

linear_predictions = lin_reg.predict(X_test)


# --------------------------------------------------
# Random Forest Regressor
# --------------------------------------------------

rf_reg = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

rf_reg.fit(X_train, y_train)

rf_predictions = rf_reg.predict(X_test)


# --------------------------------------------------
# Model Evaluation
# --------------------------------------------------

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


# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------

st.sidebar.header("📊 Advertising Budget")

tv = st.sidebar.number_input(
    "TV Advertising Budget",
    min_value=0.0,
    max_value=300.0,
    value=150.0,
    step=0.1
)

radio = st.sidebar.number_input(
    "Radio Advertising Budget",
    min_value=0.0,
    max_value=50.0,
    value=20.0,
    step=0.1
)

newspaper = st.sidebar.number_input(
    "Newspaper Advertising Budget",
    min_value=0.0,
    max_value=120.0,
    value=30.0,
    step=0.1
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

input_data = pd.DataFrame(
    {
        "TV": [tv],
        "Radio": [radio],
        "Newspaper": [newspaper]
    }
)


if st.button("🔮 Predict Sales"):

    prediction = rf_reg.predict(input_data)[0]

    st.success(
        f"📈 Predicted Sales: **{prediction:.2f} thousand units**"
    )


# --------------------------------------------------
# Model Information
# --------------------------------------------------

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


# --------------------------------------------------
# Model Comparison
# --------------------------------------------------

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


# --------------------------------------------------
# Feature Importance
# --------------------------------------------------

st.subheader("📌 Advertising Channel Impact")

importance = pd.DataFrame(
    {
        "Advertising Channel": [
            "TV",
            "Radio",
            "Newspaper"
        ],
        "Importance": rf_reg.feature_importances_
    }
)

importance["Importance"] = (
    importance["Importance"] * 100
).round(2)

st.bar_chart(
    importance.set_index("Advertising Channel")
)


# --------------------------------------------------
# Dataset Information
# --------------------------------------------------

st.markdown("---")

st.subheader("📚 Dataset Information")

st.write(
    "The dataset contains 200 markets with advertising "
    "budgets for TV, Radio, and Newspaper and the "
    "corresponding product sales."
)

st.write(
    "The original project uses an 80/20 train-test split "
    "and compares Linear Regression with Random Forest "
    "Regression."
)

st.caption(
    "OIBSIP Data Science Internship — Sales Prediction"
)
