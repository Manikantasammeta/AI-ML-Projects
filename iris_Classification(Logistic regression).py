import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,log_loss
from sklearn.linear_model import LogisticRegression
import streamlit as st



def load_dataset():
    data=sns.load_dataset('iris')
    df = pd.DataFrame(data)
    print(df)
    return df


def Data_Details(data):
    print('Data Details')
    print('Top 5 Rows \n',data.head())
    print(f"\nDataset Shape: {data.shape}")
    print("\nStatistics:")
    print(data.describe())
    print("\nClass Distribution:")
    print(data['species'].value_counts())
    print()
    print("\nData Information:")
    print(data.info())

def Data_Preprocessing(data):
    if data.isna().sum().sum() > 0:
        data=data.fillna(data.mean())
    
        
    if data.isnull().sum().sum() > 0:
        data=data.dropna()
        
    if data.duplicated().sum() > 0:
        data=data.drop_duplicates()
        
    return data

def Data_Visualization(data):

    plt.figure(figsize=(15, 10))
    sns.pairplot(data, hue='species', palette='viridis')
  
    
    numeric_data = data.drop('species', axis=1)
    sns.heatmap(numeric_data.corr(), annot=True)
    plt.title("Correlation Heatmap")
 


def Data_Encoding(data):
    label_encoder = LabelEncoder()
    data['species'] = label_encoder.fit_transform(data['species'])
    return data

def data_split(data):
    X = data.drop('species', axis=1)
    y = data['species']
    
    X_temp, X_check, y_temp, y_check = train_test_split(X, y, test_size=0.1)

    X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, train_size=0.8)
    print("70% of data for training",X_train.shape)
    print("20% of data for testing",X_test.shape)
    print("10% of data for validation",X_check.shape)
    
    return X_train, X_test, X_check, y_train, y_test, y_check


def data_StandardScaler(X_train, X_test, X_check):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
    X_test_scaled = pd.DataFrame(scaler.transform(X_test))
    X_check_scaled = pd.DataFrame(scaler.transform(X_check))
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    return X_train_scaled, X_test_scaled, X_check_scaled


def Model_Training(X_train_scaled ,y_train):
    model=LogisticRegression()
    model.fit(X_train_scaled,y_train)
    return model

def Model_Evaluation(model, X_test_scaled, y_test):
    # Use .values consistently to avoid warnings
    y_pred = model.predict( X_test_scaled)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Model Score:", model.score( X_test_scaled, y_test))
    print("Log Loss:", log_loss(y_test, model.predict_proba( X_test_scaled), labels=[0, 1, 2]))

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))  
def Model_Prediction(model, X_check_scaled, y_check):
    
    y_pred = model.predict(X_check_scaled)
    print("Actual:\n",np.array(y_check))
    print("Predictions:\n", y_pred)
    accuracy = accuracy_score(y_check, y_pred)
    print("Accuracy:", accuracy)
    print(" Model Score:", model.score(X_check_scaled, y_check))
    print("\n Classification Report:\n", classification_report(y_check, y_pred))
    
def Save_Model(model):
    with open('my_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    print("Model saved to my_model.pkl")

    
    
def main():
    df=load_dataset()
    Data_Details(df)
    df=Data_Preprocessing(df)
    Data_Visualization(df)
    df=Data_Encoding(df)
    X_train, X_test, X_check, y_train, y_test, y_check=data_split(df)
    X_train_scaled, X_test_scaled, X_check_scaled=data_StandardScaler(X_train, X_test, X_check)
    model=Model_Training(X_train_scaled, y_train)
    Model_Evaluation(model,X_test_scaled,y_test)
    Model_Prediction(model,X_check_scaled,y_check)
    Save_Model(model)
    
    

if __name__ == "__main__":
    main()

    
    
    
    
    
    
