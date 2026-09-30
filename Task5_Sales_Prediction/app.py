import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Page settings
st.set_page_config(
    page_title="Sales Prediction",
    page_icon="📈",
    layout="centered"
)


# Title
st.title("📈 Sales Prediction")

st.write(
    "Enter the advertising budget for TV, Radio, and Newspaper "
    "to predict product sales using Machine Learning."
)

st.markdown("---")


# Load dataset
df = pd.read_csv(
    "Task5_Sales_Prediction/Advertising.csv",
    index_col=0
)


# Remove duplicate rows
df = df.drop_duplicates().reset_index(drop=True)


# Features and target
X = df[
    [
        "TV",
        "Radio",
        "Newspaper"
    ]
]

y = df["Sales"]


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
st.subheader("💰 Enter Advertising Budget")


# TV advertising
tv = st.number_input(
    "TV Advertising Budget",
    min_value=0.0,
    max_value=500.0,
    value=100.0,
    step=0.1
)


# Radio advertising
radio = st.number_input(
    "Radio Advertising Budget",
    min_value=0.0,
    max_value=100.0,
    value=20.0,
    step=0.1
)


# Newspaper advertising
newspaper = st.number_input(
    "Newspaper Advertising Budget",
    min_value=0.0,
    max_value=100.0,
    value=20.0,
    step=0.1
)


# Prediction button
if st.button("🔮 Predict Sales"):

    # Create input dataframe
    input_data = pd.DataFrame(
        {
            "TV": [tv],
            "Radio": [radio],
            "Newspaper": [newspaper]
        }
    )


    # Predict using Random Forest
    prediction = rf_model.predict(
        input_data
    )[0]


    # Display result
    st.markdown("---")

    st.subheader("📈 Prediction Result")

    st.success(
        f"Predicted Sales: **{prediction:.2f} units**"
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
    "The dataset contains advertising expenditure for "
    "TV, Radio, and Newspaper along with the resulting Sales."
)

st.write(
    "The application uses an 80/20 train-test split "
    "and compares Linear Regression with Random Forest "
    "Regression."
)


# Footer
st.markdown("---")

st.caption(
    "OIBSIP Data Science Internship — Task 5"
)
