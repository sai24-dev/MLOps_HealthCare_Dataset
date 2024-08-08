#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


# Problem Statement: Data with health care attributes and other personal details to classify if the person gets stroke or not based on their health atrributes

# # Data Loading and Analysis
# 

# In[2]:


data = pd.read_csv('/workspaces/MLOps_HealthCare_Dataset/Simple_Deployment/healthcare-dataset.csv')
data.describe()


# In[3]:


data.info()


# ** Few Early Observations**
# 
# 1. There are 3 binary featuers - {hypertension,heart_disease,stroke'}
# 2. There are 2 numerical Features which needs normalization('Bmi, glucose_level and age')
# 3. We need to convert categorical variables to numerical values.
# 4. Features are less. Need to create custom featurs.

# In[4]:


#looking for some basic data discrepency
display(data.isnull().sum())
display(data.shape)


# In[5]:


plt.style.use('ggplot')

plots = ['age', 'avg_glucose_level', 'bmi']

plt.figure(figsize=(15, 5))  # Adjust figsize to fit all plots comfortably

for i, column in enumerate(plots):
    plt.subplot(1, 3, i+1)
    sns.histplot(data[column], color='black', bins='auto', kde=True)
    plt.title(column)
    plt.grid(True)

plt.tight_layout()  
plt.show()


# **The distribution of values of avg-glucose_level is not a normal distribution**
# 
# **avg_glucose_level seems to be skewed plus is multimodel distribution so MinMaxScaler and StandardScaler might not work here**

# **No duplicate values in the data**
# 
# **For missing values in the BMI imputation, going to use  mean  value  for imputation as the distribution of the bmi data seems to be a normal distribution**

# In[6]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


# In[7]:


#1st impute/fill the missing values
#only BMI has missing values 
imputer = SimpleImputer(strategy = 'mean')
data['bmi']=imputer.fit_transform(data[['bmi']])
encoded_data= data.copy()
#2nd Scale down the numerical features
features_to_scale=['age','bmi']
scaler = MinMaxScaler()
encoded_data[features_to_scale]=scaler.fit_transform(encoded_data[features_to_scale])


# In[8]:


#as the 'avg glucose level dosent have a normal distribution hence i am usign QuantileTransformer here 
from sklearn.preprocessing import QuantileTransformer

# Initialize QuantileTransformer
scaler = QuantileTransformer(output_distribution='uniform')

# Apply quantile transformation to avg_glucose_level
encoded_data['avg_glucose_level'] = scaler.fit_transform(encoded_data[['avg_glucose_level']])


# In[9]:


df = encoded_data.copy()
df.columns


# In[10]:


# List of columns to one-hot encode
columns_to_encode = ['Residence_type', 'work_type', 'smoking_status','ever_married','gender']

# Iterate through each column and apply pd.get_dummies
for column in columns_to_encode:
    encoded_column = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df, encoded_column], axis=1)
    df = df.drop(columns=[column],axis=1)

# Convert boolean to integers
df = df.astype(int)


# In[11]:


df.drop('id',axis=1,inplace=True)


# **The data now has around 23 features **
# 
# **The dataframe 'df' now has clean data**

# In[12]:


df.drop(['Residence_type_Rural','work_type_Private','smoking_status_Unknown', 'smoking_status_formerly smoked',
         'ever_married_Yes', 'gender_Male'], axis=1, inplace=True)


# In[13]:


df.columns


# # Data Split

# In[14]:


X = df.drop('stroke',axis=1)
y = df['stroke']
X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size = 0.2,random_state=42) 


# # Model Selection

# **Train Logistic Regression Classifier**

# In[15]:


# Instantiate Logistic Regression model
logreg = LogisticRegression()

# Train the model
logreg.fit(X_train, y_train)


# Predict the model output

# In[16]:


# Predict on validation set
y_pred = logreg.predict(X_valid)

# Calculate accuracy
accuracy = accuracy_score(y_valid, y_pred)
print(f'Accuracy: {accuracy}')


# # Model Training and Hyperparameter Tuning

# In[17]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Initialize models
xgb = XGBClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()

# Define parameter grid
param_grid_xgb = {'n_estimators': [100, 200], 'max_depth': [3, 6, 9]}
param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [3, 6, 9]}
param_grid_lr = {'C': [0.1, 1, 10]}

# Grid search
grid_search_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5)
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5)
grid_search_lr = GridSearchCV(lr, param_grid_lr, cv=5)

# Fit models
grid_search_xgb.fit(X_train, y_train)
grid_search_rf.fit(X_train, y_train)
grid_search_lr.fit(X_train, y_train)


# #  Model Evaluation

# In[23]:


from sklearn.metrics import accuracy_score

# Evaluate models
y_pred_xgb = grid_search_xgb.predict(X_valid)
y_pred_rf = grid_search_rf.predict(X_valid)
y_pred_lr = grid_search_lr.predict(X_valid)

print(f'XGBoost Accuracy: {accuracy_score(y_valid, y_pred_xgb)}')
print(f'Random Forest Accuracy: {accuracy_score(y_valid, y_pred_rf)}')
print(f'Logistic Regression Accuracy: {accuracy_score(y_valid, y_pred_lr)}')


# In[25]:


from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn


# Start MLflow experiment
mlflow.set_experiment("Healthcare_Model_Experiment")


# Initialize models
xgb = XGBClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()

# Define parameter grid
param_grid_xgb = {'n_estimators': [100, 200], 'max_depth': [3, 6, 9]}
param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [3, 6, 9]}
param_grid_lr = {'C': [0.1, 1, 10]}

# Grid search
grid_search_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5)
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5)
grid_search_lr = GridSearchCV(lr, param_grid_lr, cv=5)

# Fit models
grid_search_xgb.fit(X_train, y_train)
grid_search_rf.fit(X_train, y_train)
grid_search_lr.fit(X_train, y_train)

with mlflow.start_run():
    # Evaluate models
    y_pred_xgb = grid_search_xgb.predict(X_valid)
    y_pred_rf = grid_search_rf.predict(X_valid)
    y_pred_lr = grid_search_lr.predict(X_valid)

    print(f'XGBoost Accuracy: {accuracy_score(y_valid, y_pred_xgb)}')
    mlflow.log_metric("XGBoost accuracy", {accuracy_score(y_valid, y_pred_xgb)})
    print(f'Random Forest Accuracy: {accuracy_score(y_valid, y_pred_rf)}')
    mlflow.log_metric("Random Forest Accuracy",{accuracy_score(y_valid, y_pred_rf)})
    print(f'Logistic Regression Accuracy: {accuracy_score(y_valid, y_pred_lr)}')
    mlflow.log_metric("Logistic Regression Accuracy",{accuracy_score(y_valid, y_pred_lr)})


# # Save the Best Model

# In[20]:


import joblib

# Save the best model
best_model = grid_search_lr.best_estimator_
joblib.dump(best_model, 'best_model.pkl')


# # MLflow Tracking

# In[21]:


import mlflow
import mlflow.sklearn

# Start MLflow run
with mlflow.start_run():
    mlflow.log_param('Model', 'lr')
    mlflow.log_param('Best Params', grid_search_lr.best_params_)
    mlflow.log_metric('Accuracy', accuracy_score(y_valid, y_pred_lr))
    mlflow.sklearn.log_model(best_model, 'model')

