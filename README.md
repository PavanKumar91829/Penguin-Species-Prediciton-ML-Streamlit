# 🐧 Penguin Species Prediction

A machine learning project that predicts penguin species based on physical measurements using K-Nearest Neighbors (KNN) classification algorithm.

## Project Overview

This project uses the Palmer Penguins dataset to build a machine learning model that can classify penguins into three species (Adelie, Chinstrap, and Gentoo) based on their physical characteristics. The project includes:

1. **Traditional Machine Learning Implementation**: A Python script that demonstrates the complete machine learning workflow from data preprocessing to model evaluation.

2. **Interactive Web Application**: A Streamlit-based web application that allows users to input penguin measurements and receive species predictions in real-time.

## Dataset

The dataset (`penguins_size.csv`) contains measurements for 344 penguins from three different species:

- **Adelie**: 152 samples
- **Chinstrap**: 68 samples
- **Gentoo**: 124 samples

### Features:

- **species**: Target variable (Adelie, Chinstrap, Gentoo)
- **island**: Island where the penguin was observed (Biscoe, Dream, Torgersen)
- **culmen_length_mm**: Culmen length in millimeters
- **culmen_depth_mm**: Culmen depth in millimeters
- **flipper_length_mm**: Flipper length in millimeters
- **body_mass_g**: Body mass in grams
- **sex**: Sex of the penguin (MALE, FEMALE)

## Project Files

### 1. Traditional Machine Learning Script

The `Penguin_Species_Prediction.py` file contains the complete machine learning workflow:

- Data loading and exploration
- Data preprocessing (handling missing values, encoding categorical variables)
- Feature scaling
- Model training (KNN Classifier)
- Model evaluation
- Example predictions

This script is ideal for understanding the machine learning concepts and workflow used in this project.

### 2. Streamlit Web Application

The `Streamlit/Penguin_Species_Prediction_KNN_Classifier.py` file implements an interactive web application:

- User interface for inputting penguin measurements
- Real-time prediction of penguin species
- Integration with the trained KNN model
- Interactive elements for a better user experience

## Model

The project uses a **K-Nearest Neighbors (KNN) Classifier** with the following characteristics:

- **n_neighbors**: 4
- **Features**: All physical measurements plus encoded categorical variables
- **Preprocessing**: 
  - Standard scaling for numerical features
  - Label encoding for island
  - One-hot encoding for sex
- **Performance**:
  - Train/test split: 80/20

## Installation and Usage

### Prerequisites

- Python 3.7+
- Required packages: streamlit, pandas, numpy, scikit-learn

### Running the Traditional ML Script

```
python Penguin_Species_Prediction.py
```

### Running the Streamlit Application

```
cd Streamlit
streamlit run Penguin_Species_Prediction_KNN_Classifier.py
```

The Streamlit application will open in your web browser, allowing you to:
1. Input penguin measurements (culmen length, culmen depth, flipper length, body mass)
2. Select the island where the penguin was observed
3. Select the penguin's sex
4. Get a prediction of the penguin's species

## Project Structure

```
├── penguins_size.csv          # Dataset file
├── Penguin_Species_Prediction.py  # Traditional ML implementation
├── Streamlit/                 # Streamlit application directory
│   └── Penguin_Species_Prediction_KNN_Classifier.py  # Web application
└── README.md                  # Project documentation
```

## Technologies Used

- **Python**: Programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and preprocessing
- **Streamlit**: Web application framework

## How It Works

1. **Data Preprocessing**:
   - Missing values are removed
   - Categorical variables (island, sex) are encoded
   - Numerical features are scaled

2. **Model Training**:
   - The KNN classifier is trained on the preprocessed data
   - The model learns to associate penguin measurements with species

3. **Prediction**:
   - User inputs are processed in the same way as the training data
   - The model predicts the species based on the closest examples in the training data

## Future Improvements

- Implement additional machine learning models for comparison
- Add visualization of feature importance
- Include more detailed explanations of penguin species characteristics
- Deploy the application to a cloud platform for public access