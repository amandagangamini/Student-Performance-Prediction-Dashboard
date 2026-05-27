import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Page settings
st.set_page_config(
    page_title="Student Performance Dashboard",
    layout="wide"
)

# App title
st.title("Student Performance Prediction Dashboard")
st.write("This dashboard predicts student final grades using a machine learning model.")

# Load dataset
df = pd.read_csv("student-mat.csv", sep=";")

# Sidebar menu
st.sidebar.title("Dashboard Menu")
menu = st.sidebar.radio(
    "Select a section",
    ["Dataset Overview", "Data Visualization", "Prediction"]
)

# Select input features and target variable
features = ["studytime", "failures", "absences", "G1", "G2"]
target = "G3"

X = df[features]
y = df[target]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create and train machine learning model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Calculate model error
mae = mean_absolute_error(y_test, y_pred)

# -----------------------------
# Dataset Overview Page
# -----------------------------
if menu == "Dataset Overview":
    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Number of Students", df.shape[0])

    with col2:
        st.metric("Number of Columns", df.shape[1])

    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())

    with col4:
        st.metric("Duplicate Rows", df.duplicated().sum())

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Important Columns Used")
    st.write(features)

    st.subheader("Target Variable")
    st.write("G3 - Final Grade")

    st.subheader("Model Evaluation")
    st.write("Mean Absolute Error:", round(mae, 2))

    st.info(
        "Mean Absolute Error shows the average difference between actual grades and predicted grades. "
        "A lower value means better prediction."
    )

# -----------------------------
# Data Visualization Page
# -----------------------------
elif menu == "Data Visualization":
    st.header("Data Visualization")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Study Time vs Final Grade")
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        ax1.scatter(df["studytime"], df["G3"])
        ax1.set_xlabel("Study Time")
        ax1.set_ylabel("Final Grade")
        ax1.set_title("Study Time vs Final Grade")
        st.pyplot(fig1, use_container_width=False)

    with col2:
        st.subheader("Absences vs Final Grade")
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.scatter(df["absences"], df["G3"])
        ax2.set_xlabel("Absences")
        ax2.set_ylabel("Final Grade")
        ax2.set_title("Absences vs Final Grade")
        st.pyplot(fig2, use_container_width=False)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("G1 vs Final Grade")
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        ax3.scatter(df["G1"], df["G3"])
        ax3.set_xlabel("G1 Grade")
        ax3.set_ylabel("Final Grade")
        ax3.set_title("G1 vs Final Grade")
        st.pyplot(fig3, use_container_width=False)

    with col4:
        st.subheader("G2 vs Final Grade")
        fig4, ax4 = plt.subplots(figsize=(4, 3))
        ax4.scatter(df["G2"], df["G3"])
        ax4.set_xlabel("G2 Grade")
        ax4.set_ylabel("Final Grade")
        ax4.set_title("G2 vs Final Grade")
        st.pyplot(fig4, use_container_width=False)

    st.subheader("Actual vs Predicted Grades")

    comparison_df = pd.DataFrame({
        "Actual Grade": y_test.values,
        "Predicted Grade": y_pred
    })

    st.dataframe(comparison_df.head(10))

    fig5, ax5 = plt.subplots(figsize=(4, 3))
    ax5.plot(comparison_df["Actual Grade"].values, label="Actual Grade")
    ax5.plot(comparison_df["Predicted Grade"].values, label="Predicted Grade")
    ax5.set_xlabel("Student Number")
    ax5.set_ylabel("Grade")
    ax5.set_title("Actual vs Predicted Grades")
    ax5.legend()
    st.pyplot(fig5, use_container_width=False)

# -----------------------------
# Prediction Page
# -----------------------------
elif menu == "Prediction":
    st.header("Predict Student Final Grade")

    st.write("Enter student details below and click the prediction button.")

    studytime = st.slider(
        "Weekly Study Time",
        min_value=1,
        max_value=4,
        value=2
    )

    failures = st.number_input(
        "Number of Past Failures",
        min_value=0,
        max_value=4,
        value=0
    )

    absences = st.number_input(
        "Number of Absences",
        min_value=0,
        max_value=100,
        value=4
    )

    G1 = st.slider(
        "First Period Grade - G1",
        min_value=0,
        max_value=20,
        value=10
    )

    G2 = st.slider(
        "Second Period Grade - G2",
        min_value=0,
        max_value=20,
        value=10
    )

    if st.button("Predict Final Grade"):
        input_data = pd.DataFrame({
            "studytime": [studytime],
            "failures": [failures],
            "absences": [absences],
            "G1": [G1],
            "G2": [G2]
        })

        prediction = model.predict(input_data)

        st.success(f"Predicted Final Grade: {round(prediction[0], 2)}")