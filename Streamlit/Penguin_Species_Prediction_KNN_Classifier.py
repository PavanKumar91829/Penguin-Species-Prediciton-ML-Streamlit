import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# Set up Streamlit page
st.set_page_config(page_title="Penguin Species Predictor", layout="centered")
st.title("🐧 Penguin Species Predictor")


# Load and preprocess data
df = pd.read_csv('penguins_size.csv')

# Drop rows with missing values
df = df.dropna()

# Remove invalid 'sex' entries
df = df[df['sex'] != '.']

# One-hot encode 'sex' column (creates 'sex_FEMALE' and 'sex_MALE')
df = pd.get_dummies(df, columns=['sex'])

# Label encode 'island' column
le = LabelEncoder()
df['island_encode'] = le.fit_transform(df['island'])

# Drop the original 'island' column
df.drop('island', axis=1, inplace=True)

# Split features and target
X = df.drop('species', axis=1)
y = df['species']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)


# Train KNN model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)

# Evaluate the model
train_acc = accuracy_score(y_train, knn.predict(X_train))
test_acc = accuracy_score(y_test, knn.predict(X_test))


# User Input Section

st.subheader("Enter Penguin Features:")

culmen_length = st.number_input("1. Culmen Length (mm)", min_value=30.0, max_value=70.0, step=0.1)
culmen_depth = st.number_input("2. Culmen Depth (mm)", min_value=13.0, max_value=25.0, step=0.1)
flipper_length = st.number_input("3. Flipper Length (mm)", min_value=170, max_value=240, step=1)
body_mass = st.number_input("4. Body Mass (g)", min_value=2500, max_value=6500, step=10)

island = st.selectbox("5. Island", options=["Biscoe", "Dream", "Torgersen"])
sex = st.selectbox("6. Sex", options=["MALE", "FEMALE"])  # Capitalized to match column names

# Encode and Scale Inputs
island_encoded = le.transform([island])[0]
sex_MALE = 1 if sex == "MALE" else 0
sex_FEMALE = 1 if sex == "FEMALE" else 0

# Create a DataFrame for prediction input
input_df = pd.DataFrame([{
    'culmen_length_mm': culmen_length,
    'culmen_depth_mm': culmen_depth,
    'flipper_length_mm': flipper_length,
    'body_mass_g': body_mass,
    'sex_FEMALE': sex_FEMALE,
    'sex_MALE': sex_MALE,
    'island_encode': island_encoded
}])

# Scale user input using the same scaler
input_scaled = scaler.transform(input_df)
input_scaled_df = pd.DataFrame(input_scaled, columns=input_df.columns)


# Prediction
if st.button("Predict Species"):
    prediction = knn.predict(input_scaled_df)
    st.success(f"The predicted penguin species is: **{prediction[0]}**")
