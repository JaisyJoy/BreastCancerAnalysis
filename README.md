## Breast Cancer Prediction Project


### Overview


This project aims to build a predictive model to classify whether a patient has breast cancer based on various features. The project includes data preparation, model training, and a prediction pipeline using machine learning techniques.

### Project Structure


- `dataprep.ipynb`: Jupyter notebook for data preprocessing.
- `app.py`: Python script for running the Streamlit web application.
- `prediction.ipynb`: Jupyter notebook for generating predictions using the trained model.
- `breast_cancer_model.pkl`: Trained machine learning model.
- `scaler.pkl`: Scaler object used to standardize the features.
- `selected_columns.pkl`: List of selected columns/features used in the model.
- `selector.pkl`: Column selector for preprocessing.
- `requirements.txt`: List of Python dependencies required for the project.

### Setup Instructions

1. Install Dependencies
Make sure you have Python installed. You can install the required Python packages using pip:

```bash
pip install -r requirements.txt
```bash


The requirements.txt file includes the following libraries:

-`pandas`
-`numpy`
-`matplotlib`
-`scikit-learn`
-`streamlit`

2. Data Preparation
Open and run the dataprep.ipynb notebook to preprocess the data. This step involves:

Loading the dataset.
Cleaning the data (handling missing values, outliers, etc.).
Feature engineering (scaling, selection of important features, etc.).
Saving the processed data and objects like the scaler and selected columns for later use.

3. Model Training
Training of the model is presumably done in a separate notebook or file (this file is not provided, but likely related to the data preprocessing notebook). This step involves:

Splitting the data into training and testing sets.
Training a machine learning model (e.g., logistic regression, random forest) using the scikit-learn library.
Saving the trained model as breast_cancer_model.pkl.

4. Prediction Pipeline
Open and run the prediction.ipynb notebook to generate predictions. This step includes:

Loading the preprocessed data.
Loading the trained model (breast_cancer_model.pkl).
Scaling the new data using scaler.pkl.
Selecting the relevant columns using selected_columns.pkl.
Making predictions and evaluating model performance.

5. Running the Web Application
You can run the Streamlit web application to interact with the model via a simple interface. The application allows users to input new data and get predictions in real-time.

To run the app, use the following command:

```bash
streamlit run app.py
```bash
Once the app is running, you can access it in your web browser by navigating to:
```bash
http://localhost:8506/
```bash
![image](https://github.com/user-attachments/assets/99f1c14b-a763-496c-abd2-129e7603f873)


This will open the Streamlit web app where you can interact with the prediction model.

### Conclusion
This project provides a complete pipeline for building, training, and deploying a breast cancer prediction model. The Jupyter notebooks allow for easy experimentation and analysis, while the Streamlit app provides an intuitive interface for end-users.

















