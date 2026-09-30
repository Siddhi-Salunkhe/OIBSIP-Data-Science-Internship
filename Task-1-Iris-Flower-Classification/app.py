
import streamlit as st
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🌸 Iris Flower Classification")

st.write(
    "Predict the species of an Iris flower using "
    "Machine Learning."
)

st.markdown("---")


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target


# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# Feature Scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# Train Model
# --------------------------------------------------

model = LogisticRegression(
    max_iter=200,
    random_state=42
)

model.fit(X_train_scaled, y_train)


# --------------------------------------------------
# Model Accuracy
# --------------------------------------------------

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("🌼 Flower Measurements")

sepal_length = st.sidebar.slider(
    "Sepal Length (cm)",
    min_value=4.0,
    max_value=8.0,
    value=5.8,
    step=0.1
)

sepal_width = st.sidebar.slider(
    "Sepal Width (cm)",
    min_value=2.0,
    max_value=4.5,
    value=3.0,
    step=0.1
)

petal_length = st.sidebar.slider(
    "Petal Length (cm)",
    min_value=1.0,
    max_value=7.0,
    value=4.0,
    step=0.1
)

petal_width = st.sidebar.slider(
    "Petal Width (cm)",
    min_value=0.1,
    max_value=2.5,
    value=1.2,
    step=0.1
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

input_data = pd.DataFrame(
    [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]],
    columns=iris.feature_names
)

input_scaled = scaler.transform(input_data)


if st.button("🔮 Predict Iris Species"):

    prediction = model.predict(input_scaled)[0]

    probabilities = model.predict_proba(input_scaled)[0]

    predicted_species = iris.target_names[prediction]

    st.success(
        f"🌸 Predicted Species: **{predicted_species.title()}**"
    )

    st.markdown("### Prediction Probability")

    probability_df = pd.DataFrame(
        {
            "Species": [
                species.title()
                for species in iris.target_names
            ],
            "Probability": probabilities
        }
    )

    probability_df["Probability"] = (
        probability_df["Probability"] * 100
    ).round(2)

    st.dataframe(
        probability_df,
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# Model Information
# --------------------------------------------------

st.markdown("---")

st.subheader("📊 Model Information")

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


# --------------------------------------------------
# Dataset Information
# --------------------------------------------------

st.markdown("---")

st.subheader("📚 Dataset Information")

st.write(
    "The Iris dataset contains 150 samples belonging "
    "to three species: Iris Setosa, Iris Versicolor, "
    "and Iris Virginica."
)

st.write(
    "The model uses four features: sepal length, "
    "sepal width, petal length, and petal width."
)

st.caption(
    "OIBSIP Data Science Internship — Task 1"
)
