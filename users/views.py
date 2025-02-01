# Create your views here.
from ast import alias
from concurrent.futures import process
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages

import Patient_Health_Analysis

from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import datetime as dt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# Create your views here.


def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not Yet Activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):

    return render(request, 'users/UserHomePage.html', {})
#===========================================================================



def DatasetView(request):
    path = settings.MEDIA_ROOT + "//" + 'patients.csv'
    df = pd.read_csv(path, nrows=8000)
    df = df.to_html
    return render(request, 'users/viewdataset.html', {'data': df})


#===============================================================================

import pandas
from django.shortcuts import render
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


path=settings.MEDIA_ROOT + "//" + 'patients.csv'
data=pandas.read_csv(path)
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
data['Group']=lb.fit_transform(data['Group'])
data['gender']=lb.fit_transform(data['gender'])
data['Hyper Tensive']=lb.fit_transform(data['Hyper Tensive'])
data['Atrial Fibrillation']=lb.fit_transform(data['Atrial Fibrillation'])
data['CHD with no MI']=lb.fit_transform(data['CHD with no MI'])
data['Diabetes']=lb.fit_transform(data['Diabetes'])
data['Deficiency Anemias']=lb.fit_transform(data['Deficiency Anemias'])
data['Depression']=lb.fit_transform(data['Depression'])
data['Hyperlipemia']=lb.fit_transform(data['Hyperlipemia'])
# data['average study time 1-2']=lb.fit_transform(data['average study time 1-2'])
# data['attendance.1']=lb.fit_transform(data['attendance.1'])
# data['2-1 mid']=lb.fit_transform(data['2-1 mid'])
data=data[['age','gender','BMI','Hyper Tensive','Diabetes','Depression','Heart Rate','Respiratory rate','Temperature','RBC','Outcome']]
x=data.iloc[:,0:-1]
y=data.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

def ml(request):
    # Initialize classifiers
    classifiers = {
        'Random Forest Classifier': RandomForestClassifier(),
        'Ridge Classifier': RidgeClassifier(),
        'Extra Trees Classifier': ExtraTreesClassifier(),
        'Logistic Regression': LogisticRegression(),
        'Gradient Boosting Classifier': GradientBoostingClassifier(),
        'AdaBoost Classifier': AdaBoostClassifier(),
        'K Neighbors Classifier': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Decision Tree Classifier': DecisionTreeClassifier(),
        'XGBoost Classifier': XGBClassifier()
    }

    results = {}

    # Fit and evaluate each classifier
    for clf_name, clf in classifiers.items():
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Store results
        results[clf_name] = {
            'accuracy': acc,
            'classification_report': classification_rep,
            'confusion_matrix': cm
        }

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {clf_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'confusion_matrix_{clf_name}.png')  # Save the plot
        plt.show()

    return render(request, 'users/ml.html', {'results': results})


def predictTrustWorthy(request):
    import pandas as pd
    from xgboost import XGBClassifier

    xgb_model = XGBClassifier()
    xgb_model.fit(x_train,y_train)

    if request.method == 'POST':
        # Extracting data from the POST request
        age = request.POST.get("age")
        gender = request.POST.get("gender")
        BMI = request.POST.get("bmi")
        Hyper_Tensive = request.POST.get("hypertensive")
        Diabetes = request.POST.get("Diabetes")
        Depression = request.POST.get("Depression")
        Heart_Rate = request.POST.get("heartRate")
        Respiratory_rate = request.POST.get("respiratoryRate")
        Temperature = request.POST.get("temperature")
        RBC = request.POST.get("rbc")

        user_input = [age, gender, BMI, Hyper_Tensive, Diabetes, Depression, Heart_Rate, Respiratory_rate, Temperature, RBC]
        feat_list = np.array(user_input, dtype=object)
        print(user_input)
        y_pred = xgb_model.predict([feat_list])

        if y_pred[0] == 1:
            msg = 'Outcome(Patient health condition is normal)'
        else:
            msg = 'Outcome(Patient health status is abnormal or critical)'
        return render(request, "users/predictionForm.html", {'msg': msg})
    else:
        return render(request, "users/predictionForm.html", {})


