# Patients Health Analysis using Machine Learning

## Project Overview
This project explores the application of machine learning techniques in analyzing patient health data to improve healthcare outcomes. The primary objective is to develop a predictive model that can forecast patient health conditions based on a variety of input features such as heart rate, body mass index (BMI), and temperature.

## Table of Contents
1. [Introduction](#Introduction)
2. [Data Description](#Data-Description)
3. [Project Structure](#Project-Structure)
4. [Installation](#Installation)
5. [Data Preprocessing](#Data-Preprocessing)
6. [Machine Learning](#Machine-Learning)
7. [Classifiers](#Classifiers)
8. [Prediction](#Prediction)
9. [Results](#Results)

## Introduction
As technology advances, vast improvements have been made in the medical field. It is important to know the health status of the patient. Proper analysis is needed to find the root cause of health issues. Many researchers use their own methods to assess patient health. This project demonstrates how to analyze patient health using the latest data science and machine learning tools to provide insights from that analysis.

## Data Description
The data is collected from HealthData.gov, a U.S. government health data repository. The dataset includes:
- Age
- Gender
- Basic health readings

## Project Structure
- **Home**
- **User**
  - Data View
  - [ML](#Machine-Learning)
    - Data Preprocessing
    - Train
    - Test
- **Prediction**
- **Admin**
  - User Details
- **Register**
  - User Registration Form


## Technologies Used

- **Backend**: Django, Python, ML
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Database**: SQLite (development), PostgreSQL/MySQL (production)
- **Deployment**: Docker, Nginx, Gunicorn, AWS/GCP


## Installation

### Prerequisites

- Python 3.x
- pip
- virtualenv (optional but recommended)

### Steps

1. **Clone the repository:**

   ```bash
   git clone ttps://github.com/ujen42/PatientCareOptimization.git

2. **Create a virtual environment (optional):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   
4. **Change directory:**
   ```bash
   cd Patient_Health_Analysis

5. **Apply migrations:**
   ```bash
   python manage.py migrate

6. **Create a superuser (optional, for admin access):**
   ```bash
   python manage.py createsuperuser

7. **Run the development server:**
   ```bash
   python manage.py runserver

 
## Data-Preprocessing
- Data Collection
- filter data
  - filter data based on the country code, rename columns, drop unnecessary columns, and convert the DataFrame to HTML.
- Render Template

## Machine-Learning
- Data collecton
  - Imputing missing values
  - Normalizing and scaling features
  - Encoding categorical variables
  - Initializes a dictionary classifiers containing different machine learning models
- Model Training
  - Train the Classifier
  - Make Predictions
  - Calculate Accuracy
  - Generate Classification Report
  - Generate Confusion Matrix for each classifier
- Model Testing
  - Stores the accuracy
  - Plot confusion matrix
  - Save the plot
  - Display
 
## Classifiers
- RandomForestClassifier
- RidgeClassifier
- ExtraTreesClassifier
- LogisticRegression
- GradientBoostingClassifier
- AdaBoostClassifier
- KNeighborsClassifier
- GaussianNB
- DecisionTreeClassifier
- XGBClassifier

## Prediction
- Takes the data given
- Shows the condition of the patient

## Results
- **Home Page**
  ![Home Page](media/patienthome.png)
- **Admin Page**
  ![Admin Login](media/patientadminlogin.png)
  ![Admin Home](media/patientadminhome.png)
  ![User Details](media/patientuserdetails.png)
- **User Page**
  ![User Login](media/patientuserlogin.png)
  ![User Home](media/patientuserhome.png)
- **User Registration**
  ![User Register Form](media/patientuserregister.png)
- Confusion matrix
  ![AdaBoostClassifier](confusion_matrix_AdaBoostClassifier.png)
  ![DecisionTreeClassifier](confusion_matrix_DecisionTreeClassifier.png)
  ![ExtraTreesClassifier](confusion_matrix_ExtraTreesClassifier.png)
  ![GradientBoostingClassifier](confusion_matrix_GradientBoostingClassifier.png)
  ![KNeighborsClassifier](confusion_matrix_KNeighborsClassifier.png)
  ![LogisticRegression](confusion_matrix_LogisticRegression.png)
  ![NaiveBayes](confusion_matrix_NaiveBayes.png)
  ![RandomForestClassifier](confusion_matrix_RandomForestClassifier.png)
  ![RidgeClassifier](confusion_matrix_RidgeClassifier.png)
  ![XGBoostClassifier](confusion_matrix_XGBoostClassifier.png)
- classifer scores
  ![score](media/patientalgo1.png)
  ![score](media/patientalgo2.png)
  ![score](media/patientalgo3.png)
- Data set
  ![data set](media/patientdataset.png)
- **Prediction**
  ![prediction](media/patientprediction.png)




