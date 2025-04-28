import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder ,StandardScaler
from sklearn.ensemble import *
from sklearn.tree import DecisionTreeClassifier , DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression ,LogisticRegression
from sklearn.metrics import confusion_matrix ,log_loss ,classification_report ,accuracy_score
import pickle  


def data_importing():
    df=pd.read_csv('adult.data',na_values=' ?',header=None)
    return df

def data_details(df):
    print("The sahe of the dataset is ->",df.shape)
    print()
    print("The first 5 rows of the dataset are ->\n")
    print(df.head())
    print()
  
    print("The information about the dataset is ->\n")
    print(df.info())
    
    print(df.describe())
    print("The columns of the dataset are ->\n")
    print(df.columns)
    
def handle_null_values(df):
    for col in df.columns:
        if df[col].dtype == 'object':  # If the column is categorical
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:  # If the column is numerical
            df[col].fillna(df[col].mean(), inplace=True)
    return df

def data_visualization(df):
    plt.figure(figsize=(10, 6))
    df.hist(bins=30, edgecolor='black')
    plt.show()
    
    sns.countplot(x='income', data=df)
    plt.show()
    
    sns.histplot(df['age'], bins=30, kde=True)
    plt.show()
    
    sns.scatterplot(x='age', y='hours-per-week', hue='income', data=df)
    plt.show()
    
    sns.boxplot(x='hours-per-week', y='age', hue='income', data=df)
    plt.show()
    
    sns.boxplot(x='hours-per-week', y='age', hue='income', data=df, palette='Set1')
    plt.show()

    sns.boxplot(df['capital_gain'])
    plt.show()
    

def data_encoding(df):                                  # Label Encoding
    encoder= LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = encoder.fit_transform(df[col])  # Apply label encoding to the column df[col]    
    return df


def data_split(data):
    X = data.drop('income', axis=1)
    y = data['income']
    
    X_temp, X_check, y_temp, y_check = train_test_split(X, y, test_size=0.1)

    X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, train_size=0.8)
    print("70% of data for training",X_train.shape)
    print("20% of data for testing",X_test.shape)
    print("10% of data for validation",X_check.shape)
    
    return X_train, X_test, X_check, y_train, y_test, y_check



def data_standardScaler(X_train, X_test, X_check):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
    X_test_scaled = pd.DataFrame(scaler.transform(X_test))
    X_check_scaled = pd.DataFrame(scaler.transform(X_check))
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    return X_train_scaled, X_test_scaled, X_check_scaled

def model_training(X_train_scaled ,y_train):
    models=[
        RandomForestClassifier(),
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeClassifier(),
        DecisionTreeRegressor(),
        LogisticRegression()
    ]
    trained_models=[]
    
    for model in models:
        model.fit(X_train_scaled,y_train)
        trained_models.append(model)
        print(f"Trained Model: {model}\n")
        print()
        
    return trained_models

def moel_evaluation(models,X_test_scaled,y_test):
    for model in models:
        print(f"Model: {model}")
        print("Accuracy:", model.score(X_test_scaled, y_test))
        print("Log Loss:", log_loss(y_test, model.predict_proba(X_test_scaled)))
        print("Confusion Matrix:\n", confusion_matrix(y_test, model.predict(X_test_scaled)))
        print("Classification Report:\n", classification_report(y_test, model.predict(X_test_scaled)))
        print("Accuracy:", accuracy_score(y_test, model.predict(X_test_scaled)))
        print()
        
def model_prediction(models,X_check_scaled,y_check):
    for model in models:
        print(f"Model: {model}")
        print("Actual:\n",np.array(y_check))
        print("Predictions:\n", model.predict(X_check_scaled))
        print("Accuracy:", accuracy_score(y_check, model.predict(X_check_scaled)))
        print(" Model Score:", model.score(X_check_scaled, y_check))
        print("\n Classification Report:\n", classification_report(y_check, model.predict(X_check_scaled)))
        print()
    


def save_model(model):
    with open('my_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    print("Model saved to my_model.pkl")