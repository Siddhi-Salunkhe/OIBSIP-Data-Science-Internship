import streamlit as st
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Page settings
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="centered"
)


# Title
st.title("🌸 Iris Flower Classification")

st.write(
    "Enter the flower measurements below to predict "
    "the Iris flower species using Machine Learning."
)

st.markdown("---")


# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train Logistic Regression model
model = LogisticRegression(
    max_iter=200,
    random_state=42
)

model.fit(X_train_scaled, y_train)


# Model accuracy
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(
    y_test,
    y_pred
)


# Input section
st.subheader("🌼 Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:

    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1,
        step=0.1
    )

    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1
    )


with col2:

    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4,
        step=0.1
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2,
        step=0.1
    )


# Prediction button
if st.button("🔮 Predict Iris Species"):

    # Create input dataframe
    input_data = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    )


    # Scale input
    input_scaled = scaler.transform(input_data)


    # Prediction
    prediction = model.predict(input_scaled)[0]

    probabilities = model.predict_proba(input_scaled)[0]


    # Species name
    species = iris.target_names[prediction]


    # Display result
    st.markdown("---")

    st.subheader("🌸 Prediction Result")

    st.success(
        f"Predicted Iris Species: **{species.title()}**"
    )


    # Prediction probabilities
    st.subheader("📊 Prediction Probabilities")

    probability_df = pd.DataFrame(
        {
            "Species": [
                name.title()
                for name in iris.target_names
            ],
            "Probability": [
                f"{probability * 100:.2f}%"
                for probability in probabilities
            ]
        }
    )

    st.table(probability_df)


# Model information
st.markdown("---")

st.subheader("🤖 Model Information")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Model",
        "Logistic Regression"
    )

with col2:
    st.metric(
        "Test Accuracy",
        f"{accuracy * 100:.2f}%"
    )


# Dataset information
st.markdown("---")

st.subheader("📚 About the Dataset")

st.write(
    "The Iris dataset contains measurements of iris flowers "
    "including sepal length, sepal width, petal length, "
    "and petal width."
)

st.write(
    "The model classifies the flower into one of three species:"
)

st.write(
    "🌸 Setosa  |  🌸 Versicolor  |  🌸 Virginica"
)


# Footer
st.markdown("---")

st.caption(
    "OIBSIP Data Science Internship — Task 1"
)
