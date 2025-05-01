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


def data_loading():#loading the dataset
    
    df=pd.read_csv('adult.data',na_values=' ?',header=None)
    return df

def data_details(df): # displaying the details of the dataset
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
    
def handle_null_values(df): # handling the null values in the dataset
    for col in df.columns:
        if df[col].dtype == 'object':  # If the column is categorical
            df[col] = df[col].fillna(df[col].mode()[0])
        else:  # If the column is numerical
            df[col] = df[col].fillna(df[col].mean())
    print("Null values are handled")

    return df

def data_visualization(df): #data visualization of data set
    plt.figure(figsize=(10, 6))# creating a figure
    df.hist(bins=30, edgecolor='black')#plotting the histogram
    plt.show()
    
    sns.countplot(x='income', data=df)
    plt.show()
    
    sns.histplot(df['age'], bins=30, kde=True)#plotting the histogram
    plt.show()
    
    sns.scatterplot(x='age', y='hours-per-week', hue='income', data=df)#plotting the scatter plot
    plt.show()
    
    sns.boxplot(x='hours-per-week', y='age', hue='income', data=df)#plotting the box plot
    plt.show()
    
    sns.boxplot(x='hours-per-week', y='age', hue='income', data=df, palette='Set1')#plotting the box plot
    plt.show()

    sns.boxplot(df['capital_gain'])
    plt.show()
    
def adding_new_features(df):# adding new features to the dataset
 
    df['is_senior'] = df['age'] >= 60 # Creating a new feature is_senior
    df['net_capital'] = df['capital_gain'] - df['capital_loss']# Creating a new feature net_capital
    df['work_type'] = df['weekly_hours'].apply(lambda x: 'Full-time' if x >= 35 else 'Part-time')# Creating a new feature work_type

    df['is_married'] = df['marital_status'].apply(lambda x: 'Married' in x)# Creating a new feature is_married
    df['is_native_us'] = df['country_of_origin'] == 'United-States'# Creating a new feature is_native_us

  

    def categorize_education(num): # Creating a new feature education_level
        if num <= 8:
            return 'Low'
        elif 9 <= num <= 12:
            return 'Medium'
        else:
            return 'High'

    df['education_level'] = df['education_years'].apply(categorize_education)# Creating a new feature education_level

    df['experience_estimate'] = df['age'] - df['education_years']# Creating a new feature experience_estimate

    return df




def data_encoding(df):                                  # Label Encoding
    
    encoder= LabelEncoder()
    for col in df.columns:# Looping through the columns
        if df[col].dtype == 'object':
            df[col] = encoder.fit_transform(df[col])  # Apply label encoding to the column df[col]  
    print("Encoding is done")  
    return df


def data_split(df):#splitting the dataset     but here not going to use  function because we have test data set in our file named as(adult_test.data) so we can't use 
    X = df.drop('income', axis=1)
    y = df['income']
    
    X_temp, X_check, y_temp, y_check = train_test_split(X, y, test_size=0.1)

    X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, train_size=0.8)
    print("70% of data for training",X_train.shape)
    print("20% of data for testing",X_test.shape)
    print("10% of data for validation",X_check.shape)
    print("*"*100)
    return X_train, X_test, X_check, y_train, y_test, y_check



def data_standardScaler(df):#data standardization of dataset
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(df))#standardizing the data
 
    
    with open("scaler.pkl", "wb") as f:#saving the scaler in a file for future use in prediction
        pickle.dump(scaler,f)
    
    return X_train_scaled

def model_training(X_train_scaled ,y_train): #model training
    model=RandomForestRegressor()# creating a model
    model.fit(X_train_scaled,y_train)
    print("Models are trained\n",model)
    print("*"*100)
    return model

def moel_evaluation(model,X_test_scaled,y_test):#model evaluation
        print("*"*15,"Model Evalution","*"*15)
        print(f"Model: {model}")
        print("Accuracy:", model.score(X_test_scaled, y_test))
        # print("Log Loss:", log_loss(y_test, model.predict_proba(X_test_scaled)))                            getting erroe
        print("Confusion Matrix:\n", confusion_matrix(y_test, model.predict(X_test_scaled),labels = [0, 1] ) )
        print("Classification Report:\n", classification_report(y_test, model.predict(X_test_scaled)))
        print("Accuracy:", accuracy_score(y_test, model.predict(X_test_scaled)))
        print("*"*100)
        print()
        
def model_prediction(model,X_check_scaled,y_check):#model prediction this one also we are not useing because we have test data set in our file named as(adult_test.data) so we can't use
       
        print("Actual:\n",np.array(y_check))
        print("Predictions:\n", model.predict(X_check_scaled))
        are_equal = np.all(np.array(y_check) == np.array(model.predict(X_check_scaled)))
        print("All Predections are Correct...?",are_equal)  # Will print True if all are equal, False otherwise

       
        print("Accuracy:", accuracy_score(y_check, model.predict(X_check_scaled)))
        print(" Model Score:", model.score(X_check_scaled, y_check))
        print("\n Classification Report:\n", classification_report(y_check, model.predict(X_check_scaled)))
        print("*"*100)
        print()
    


def save_model(model):#model saving in a file(my_model.pkl) for future use in prediction will use it in adult_income_app.Py
    with open('my_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    print("Model saved to my_model.pkl")
    
    
def testing_model_with_adult_test_data(model): #testing the model with test data
    df=df = pd.read_csv('C:\\Users\\manik\\Downloads\\AI & ML Projects\\RandomForest\\adult\\adult_test.data',na_values=' ?',header=None,delimiter=',', skiprows=1) #loading the test data
    
    df.columns = ['age','job_type','person_weight','education_level','education_years',
            'marital_status','job_role','family_role','ethnicity','gender','capital_gain',
            'capital_loss','weekly_hours','country_of_origin','income'] #renaming the columns
    y=df['income']#separating the target variable              do all the things that we did in training data
    df=df.drop('income',axis=1)
    if df.isnull().sum().sum() > 0:
        handle_null_values(df)
    
    df=adding_new_features(df)
    if df.isnull().sum().sum() > 0:
        handle_null_values(df)
    df=data_encoding(df) 
    scaled_test_data=data_standardScaler(df)
    y=y.apply(lambda x: 0 if x == '<=50K' else 1) #converting the target variable to 0 and 1
    print('Accuracy of the model on test data:',model.score(scaled_test_data,y))
    print("Confusion Matrix:\n", confusion_matrix(y, model.predict(scaled_test_data),labels = [0, 1] ) )
    print("Classification Report:\n", classification_report(y, model.predict(scaled_test_data)))
    print("Accuracy:", accuracy_score(y, model.predict(scaled_test_data)))
    
   
def main():#main function
    df=data_loading()
    
    data_details(df)
    df.columns = ['age','job_type','person_weight','education_level','education_years',
            'marital_status','job_role','family_role','ethnicity','gender','capital_gain',
            'capital_loss','weekly_hours','country_of_origin','income']
    y=df['income'] #separating the target variable
    df=df.drop('income',axis=1)
    if df.isnull().sum().sum() > 0:
        handle_null_values(df)
   
    df=adding_new_features(df)
    if df.isnull().sum().sum() > 0:
        handle_null_values(df)
    
    #data_visualization(df)     # uncomment this line if you want to see the data visualization
    y = y.apply(lambda x: 0 if x == '<=50K' else 1)
    df=data_encoding(df)
    #X_train, X_test, X_check, y_train, y_test, y_check=data_split(df)      # suppose you guys don't have saperate test data set then go for splitting the data
    
    X_train_scaled=data_standardScaler(df) 
    
    model=model_training(X_train_scaled ,y)
    # moel_evaluation(model,X_test_scaled,y_test)    # if you going with splitting the data then uncomment this line and use this line for model evaluation
    # model_prediction(model,X_check_scaled,y_check) # if you going with splitting the data then uncomment this line and use this line for model prediction
    testing_model_with_adult_test_data(model)       # evaluating the model with test data
    
    save_model(model)#finally saving the model

    
if __name__ == "__main__":
    main()