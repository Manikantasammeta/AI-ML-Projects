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


print("Hello")
def data_loading():
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
    print("*"*100)
    
def handle_null_values(df):
    for col in df.columns:
        if df[col].dtype == 'object':  # If the column is categorical
            df[col] = df[col].fillna(df[col].mode()[0])
        else:  # If the column is numerical
            df[col] = df[col].fillna(df[col].mean())
    print("Null values are handled")

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
    
def adding_new_features(df):
 
    df['is_senior'] = df['age'] >= 60 
    df['net_capital'] = df['capital_gain'] - df['capital_loss']
    df['work_type'] = df['weekly_hours'].apply(lambda x: 'Full-time' if x >= 35 else 'Part-time')

    df['is_married'] = df['marital_status'].apply(lambda x: 'Married' in x)
    df['is_native_us'] = df['country_of_origin'] == 'United-States'

  

    def categorize_education(num):
        if num <= 8:
            return 'Low'
        elif 9 <= num <= 12:
            return 'Medium'
        else:
            return 'High'

    df['education_level'] = df['education_years'].apply(categorize_education)

    df['experience_estimate'] = df['age'] - df['education_years']

    return df




def data_encoding(df):                                  # Label Encoding
    
    encoder= LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = encoder.fit_transform(df[col])  # Apply label encoding to the column df[col]  
    print("Encoding is done")  
    return df


def data_split(df):
    X = df.drop('income', axis=1)
    y = df['income']
    
    X_temp, X_check, y_temp, y_check = train_test_split(X, y, test_size=0.1)

    X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, train_size=0.8)
    print("70% of data for training",X_train.shape)
    print("20% of data for testing",X_test.shape)
    print("10% of data for validation",X_check.shape)
    print("*"*100)
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
    model=RandomForestRegressor()
    model.fit(X_train_scaled,y_train)
    print("Models are trained\n",model)
    print("*"*100)
    return model

def moel_evaluation(model,X_test_scaled,y_test):
        print("*"*15,"Model Evalution","*"*15)
        print(f"Model: {model}")
        print("Accuracy:", model.score(X_test_scaled, y_test))
        # print("Log Loss:", log_loss(y_test, model.predict_proba(X_test_scaled)))                            getting erroe
        print("Confusion Matrix:\n", confusion_matrix(y_test, model.predict(X_test_scaled),labels = [0, 1] ) )
        print("Classification Report:\n", classification_report(y_test, model.predict(X_test_scaled)))
        print("Accuracy:", accuracy_score(y_test, model.predict(X_test_scaled)))
        print("*"*100)
        print()
        
def model_prediction(model,X_check_scaled,y_check):
       
        print("Actual:\n",np.array(y_check))
        print("Predictions:\n", model.predict(X_check_scaled))
        are_equal = np.all(np.array(y_check) == np.array(model.predict(X_check_scaled)))
        print("All Predections are Correct...?",are_equal)  # Will print True if all are equal, False otherwise

       
        print("Accuracy:", accuracy_score(y_check, model.predict(X_check_scaled)))
        print(" Model Score:", model.score(X_check_scaled, y_check))
        print("\n Classification Report:\n", classification_report(y_check, model.predict(X_check_scaled)))
        print("*"*100)
        print()
    


def save_model(model):
    with open('my_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    print("Model saved to my_model.pkl")
    
    
def testing_model_with_adult_test_data(model):
    df=pd.read_csv("adult_test.csv")   
    df.columns = ['age','job_type','person_weight','education_level','education_years',
            'marital_status','job_role','family_role','ethnicity','gender','capital_gain',
            'capital_loss','weekly_hours','country_of_origin','income']
    
    if df.isnull().sum().sum() > 0:
        handle_null_values(df)
    
    df=adding_new_features(df)
    if df.isnull().sum().sum() > 0:
        handle_null_values(df)
        
    scaled_test_data=data_standardScaler(df)
    print('Accuracy of the model on test data:',model.score())
    
   
def main():
    df=data_loading()
    
    data_details(df)
    df.columns = ['age','job_type','person_weight','education_level','education_years',
            'marital_status','job_role','family_role','ethnicity','gender','capital_gain',
            'capital_loss','weekly_hours','country_of_origin','income']
    
    print(df.columns)
    if df.isnull().sum().sum() > 0:
        handle_null_values(df)
   
    df=adding_new_features(df)
    if df.isnull().sum().sum() > 0:
        handle_null_values(df)
    
    #data_visualization(df)
    df['income'] = df['income'].apply(lambda x: 0 if x == '<=50K' else 1)
    df=data_encoding(df)
    X_train, X_test, X_check, y_train, y_test, y_check=data_split(df)
    
    X_train_scaled, X_test_scaled, X_check_scaled=data_standardScaler(X_train, X_test, X_check)
    
    model=model_training(X_train_scaled ,y_train)
    moel_evaluation(model,X_test_scaled,y_test)
    model_prediction(model,X_check_scaled,y_check)
    save_model(model)

    
if __name__ == "__main__":
    main()