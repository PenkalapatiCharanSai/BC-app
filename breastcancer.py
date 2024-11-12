import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load and prepare the dataset for initial model training
breast_cancer_dataset = load_breast_cancer()
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
data_frame['label'] = breast_cancer_dataset.target

# Select features for prediction
selected_features = ['mean radius', 'mean texture', 'mean smoothness']
X = data_frame[selected_features]
Y = data_frame['label']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, Y_train)

# Train the SVM model
svm_model = SVC()
svm_model.fit(X_train, Y_train)

# Train the Decision Tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, Y_train)

# Train the Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, Y_train)

# Streamlit app configuration
st.set_page_config(page_title="Breast Cancer Prediction App", page_icon="🎗️", layout="wide")
st.title("🎗️ Breast Cancer Prediction App")
st.markdown("""Welcome to the Breast Cancer Prediction App! This tool uses machine learning to help predict the likelihood of a breast tumor being **malignant** (cancerous) or **benign** (non-cancerous) based on specific features.""")

# Sidebar layout
st.sidebar.header("🔍 Input Features")
st.sidebar.write("Enter values for the following features, or upload a CSV file:")

# Sidebar for manual input of features
mean_radius = st.sidebar.number_input("Mean Radius", min_value=0.0, value=0.0, step=0.1)
mean_texture = st.sidebar.number_input("Mean Texture", min_value=0.0, value=0.0, step=0.1)
mean_smoothness = st.sidebar.number_input("Mean Smoothness", min_value=0.0, value=0.0, step=0.01)

# Validate feature inputs
if mean_radius <= 0 or mean_texture <= 0 or mean_smoothness <= 0:
    st.sidebar.error("Please enter positive values for all features.")

# Gather inputs into an array for prediction
input_data = [mean_radius, mean_texture, mean_smoothness]
input_data_as_numpy_array = np.array(input_data).reshape(1, -1)

# Prediction button and results display
st.write("## Prediction Results")
with st.container():
    if st.sidebar.button("🔮 Predict"):
        # Check if input values are entered
        if mean_radius <= 0 or mean_texture <= 0 or mean_smoothness <= 0:
            st.sidebar.error("Please enter valid positive values for all features or upload a CSV file.")
        else:
            # Proceed with prediction only if valid inputs are entered
            log_prediction = log_model.predict(input_data_as_numpy_array)
            svm_prediction = svm_model.predict(input_data_as_numpy_array)
            dt_prediction = dt_model.predict(input_data_as_numpy_array)
            rf_prediction = rf_model.predict(input_data_as_numpy_array)

            col1, col2, col3, col4 = st.columns(4)
            col1.success("Logistic Regression: The Breast Cancer is **Malignant** 🛑" if log_prediction[0] == 0 else "Benign ✅")
            col2.success("SVM: The Breast Cancer is **Malignant** 🛑" if svm_prediction[0] == 0 else "Benign ✅")
            col3.success("Decision Tree: The Breast Cancer is **Malignant** 🛑" if dt_prediction[0] == 0 else "Benign ✅")
            col4.success("Random Forest: The Breast Cancer is **Malignant** 🛑" if rf_prediction[0] == 0 else "Benign ✅")


# Model performance and accuracy display
st.write("## Model Performance on Test Data")
log_y_pred = log_model.predict(X_test)
svm_y_pred = svm_model.predict(X_test)
dt_y_pred = dt_model.predict(X_test)
rf_y_pred = rf_model.predict(X_test)

log_accuracy = accuracy_score(Y_test, log_y_pred)
svm_accuracy = accuracy_score(Y_test, svm_y_pred)
dt_accuracy = accuracy_score(Y_test, dt_y_pred)
rf_accuracy = accuracy_score(Y_test, rf_y_pred)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Logistic Regression Accuracy", f"{log_accuracy * 100:.2f}%")
col2.metric("SVM Accuracy", f"{svm_accuracy * 100:.2f}%")
col3.metric("Decision Tree Accuracy", f"{dt_accuracy * 100:.2f}%")
col4.metric("Random Forest Accuracy", f"{rf_accuracy * 100:.2f}%")

# Confusion matrix display for test data
st.write("### Confusion Matrices for Test Data")
for model_name, y_pred, color in zip(
    ["Logistic Regression", "SVM", "Decision Tree", "Random Forest"],
    [log_y_pred, svm_y_pred, dt_y_pred, rf_y_pred],
    ["Blues", "Greens", "Oranges", "Purples"]
):
    plt.figure(figsize=(8, 5))
    sns.heatmap(confusion_matrix(Y_test, y_pred), annot=True, fmt='d', cmap=color,
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"])
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(plt)

# Sidebar for CSV file upload
uploaded_file = st.sidebar.file_uploader("Or upload a CSV file with features", type=["csv"])
if uploaded_file is not None:
    uploaded_data = pd.read_csv(uploaded_file)
    st.sidebar.write("Sample of uploaded data:")
    st.sidebar.write(uploaded_data.head())

    # Check if required features are in the uploaded file
    if all(feature in uploaded_data.columns for feature in ['radius_mean', 'texture_mean', 'smoothness_mean']):
        # Rename columns to match model features
        X_uploaded = uploaded_data[['radius_mean', 'texture_mean', 'smoothness_mean']]
        X_uploaded.columns = selected_features
        
        # Check for NaN values
        if X_uploaded.isnull().any().any():
            st.sidebar.error("Uploaded data contains missing values. Please clean your data.")
        else:
            # Predict labels for uploaded data
            log_y_uploaded = log_model.predict(X_uploaded)
            svm_y_uploaded = svm_model.predict(X_uploaded)
            dt_y_uploaded = dt_model.predict(X_uploaded)
            rf_y_uploaded = rf_model.predict(X_uploaded)

            # Calculate accuracy for uploaded data if actual labels are provided
            if 'diagnosis' in uploaded_data.columns:
                uploaded_data['diagnosis_numeric'] = uploaded_data['diagnosis'].map({"M": 0, "B": 1})
                log_accuracy_uploaded = accuracy_score(uploaded_data['diagnosis_numeric'], log_y_uploaded)
                svm_accuracy_uploaded = accuracy_score(uploaded_data['diagnosis_numeric'], svm_y_uploaded)
                dt_accuracy_uploaded = accuracy_score(uploaded_data['diagnosis_numeric'], dt_y_uploaded)
                rf_accuracy_uploaded = accuracy_score(uploaded_data['diagnosis_numeric'], rf_y_uploaded)

                # Display accuracies for uploaded data
                st.write("## Model Performance on Uploaded Data")
                st.metric("Logistic Regression Accuracy", f"{log_accuracy_uploaded * 100:.2f}%")
                st.metric("SVM Accuracy", f"{svm_accuracy_uploaded * 100:.2f}%")
                st.metric("Decision Tree Accuracy", f"{dt_accuracy_uploaded * 100:.2f}%")
                st.metric("Random Forest Accuracy", f"{rf_accuracy_uploaded * 100:.2f}%")

                # Count malignant and benign predictions
                log_malignant_count = (log_y_uploaded == 0).sum()
                log_benign_count = (log_y_uploaded == 1).sum()
                svm_malignant_count = (svm_y_uploaded == 0).sum()
                svm_benign_count = (svm_y_uploaded == 1).sum()
                dt_malignant_count = (dt_y_uploaded == 0).sum()
                dt_benign_count = (dt_y_uploaded == 1).sum()
                rf_malignant_count = (rf_y_uploaded == 0).sum()
                rf_benign_count = (rf_y_uploaded == 1).sum()

                # Calculate percentage of each category
                total_count = len(uploaded_data)
                log_malignant_percentage = (log_malignant_count / total_count) * 100
                log_benign_percentage = (log_benign_count / total_count) * 100
                svm_malignant_percentage = (svm_malignant_count / total_count) * 100
                svm_benign_percentage = (svm_benign_count / total_count) * 100
                dt_malignant_percentage = (dt_malignant_count / total_count) * 100
                dt_benign_percentage = (dt_benign_count / total_count) * 100
                rf_malignant_percentage = (rf_malignant_count / total_count) * 100
                rf_benign_percentage = (rf_benign_count / total_count) * 100

                # Display the counts and percentages
                st.write("### Prediction Counts and Percentages for Malignant and Benign Cases")
                st.write(f"**Logistic Regression**: Malignant: {log_malignant_count} ({log_malignant_percentage:.2f}%), Benign: {log_benign_count} ({log_benign_percentage:.2f}%)")
                st.write(f"**SVM**: Malignant: {svm_malignant_count} ({svm_malignant_percentage:.2f}%), Benign: {svm_benign_count} ({svm_benign_percentage:.2f}%)")
                st.write(f"**Decision Tree**: Malignant: {dt_malignant_count} ({dt_malignant_percentage:.2f}%), Benign: {dt_benign_count} ({dt_benign_percentage:.2f}%)")
                st.write(f"**Random Forest**: Malignant: {rf_malignant_count} ({rf_malignant_percentage:.2f}%), Benign: {rf_benign_count} ({rf_benign_percentage:.2f}%)")

            # Add predictions to uploaded data
            uploaded_data['Logistic Regression Prediction'] = log_y_uploaded
            uploaded_data['SVM Prediction'] = svm_y_uploaded
            uploaded_data['Decision Tree Prediction'] = dt_y_uploaded
            uploaded_data['Random Forest Prediction'] = rf_y_uploaded

            # Display predictions in table format
            st.write("### Predictions on Uploaded Data")
            st.write(uploaded_data)

            # Provide the option to download the prediction results as CSV
            csv = uploaded_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name="breast_cancer_predictions.csv",
                mime="text/csv"
            )
