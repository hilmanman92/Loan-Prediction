import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import SelectPercentile 
from sklearn.feature_selection import f_classif, chi2
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report           
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin     

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier          

import os
import pickle
from time import time

from flask import Flask,request,app,jsonify,url_for,render_template

print('============================================')
print('============= STARTING =====================')
print('============================================')

# import dataset
print('Loading dataset...')
df = pd.read_csv('loan_data_2007_2014.csv', low_memory=False)
print('Done Loading dataset!')

# create target 
def mapping_loan_status(x):
    if x in ['Current', 'Fully Paid', 'In Grace Period', 'Does not meet the credit policy. Status:Fully Paid']:
        return 'Good'
    elif x in ['Charged Off', 'Late (31-120 days)', 'Default', 'Late (16-30 days)', 'Does not meet the credit policy. Status:Charged Off']:
        return 'Bad'
    else:
        return x

print('Mapping loan status...')
df['loan_status'] = df['loan_status'].apply(lambda x: mapping_loan_status(x))
print('Done Mapping!')

# drop unnecessary features
print('Dropping unnecessary features...')
cols_to_drop = ['Unnamed: 0', 'id', 'member_id', 'desc', 'title', 'url', 'zip_code','policy_code', 'application_type',
                'pymnt_plan', 'sub_grade', 'emp_title', 'addr_state', 'earliest_cr_line', 'initial_list_status',
                'issue_d', 'last_pymnt_d','next_pymnt_d','last_credit_pull_d']
df.drop(cols_to_drop, axis=1, inplace=True)
print('Done Dropping!')

# function to inspect missing values
def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    return pd.concat([mis_val, mis_val_percent], axis=1, keys=['Total', 'Percent'])

# condition for missing values more than 40% and under 10%
condition_1 = missing_values_table(df)['Percent'] > 40
condition_2 = (missing_values_table(df)['Percent'] > 0) & (missing_values_table(df)['Percent'] < 10)

# drop feature based on missing values
print('Dropping feature based on missing values more than 40%')
cols_to_drop = []
cols_to_drop.extend(missing_values_table(df)[condition_1].index.to_list())
df.drop(cols_to_drop, axis=1, inplace=True)
print(f'All feature successfully dropped')
print(f'Shape of dataset: {df.shape}')

# drop records based on missing values
print('Dropping records based on missing values under 10%')
cols_to_drop = []
cols_to_drop.extend(missing_values_table(df)[condition_2].index.to_list())
df.dropna(subset=cols_to_drop, inplace=True)
print(f'All missing values successfully dropped')
print(f'Shape of dataset: {df.shape}')

# inspect duplicates
print('Inspecting duplicates...')
if df.duplicated().sum().any() == 0:
    print('There is no duplicated data')
else:
    print('There is duplicated data')
    
# drop duplicates
df.drop_duplicates(inplace=True)
print(f'Shape of dataset: {df.shape}')

# function to separate categorical and numerical features
print('Starting Data Preprocessing...')
def numerical_and_categorical(df):
    numerical = []
    categorical = []
    for i in df.columns:
        if df[i].dtypes == 'int64' or df[i].dtypes == 'float64':
            numerical.append(i)
        else:
            categorical.append(i)
    print(f'Total numerical feature {len(numerical)}')
    print(f'Total categorical feature {len(categorical)}')
    return numerical, categorical

class RemoveMulticollinearity(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.to_drop_ = None
    
    def fit(self, X, y=None):
        corr_matrix = pd.DataFrame(X).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop_ = [column for column in upper.columns if any(upper[column] > self.threshold)]
        return self
    
    def transform(self, X):
        return pd.DataFrame(X).drop(columns=self.to_drop_).values
    
# class to remove multicollinearity with vif
class VIFSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=10.0):
        self.threshold = threshold
        self.features_to_keep_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.features_to_keep_ = list(X.columns)
        while True:
            vif = pd.Series(
                [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
                index=X.columns
            )
            max_vif = vif.max()
            if max_vif > self.threshold:
                feature_to_remove = vif.idxmax()
                self.features_to_keep_.remove(feature_to_remove)
                X = X.drop(columns=[feature_to_remove])
            else:
                break
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.features_to_keep_).values

# call function to separate categorical and numerical features
print('Separating categorical and numerical features...')
numerical, categorical = numerical_and_categorical(df)

# replace 'NONE' and 'ANY' with 'OTHER'
print('Replacing NONE and ANY with OTHER in home_ownership...')
df['home_ownership'] = df['home_ownership'].replace(['NONE','ANY'], 'OTHER')

# encode target variable Good as 1 and Bad as 0
print('Encoding target variable Good as 1 and Bad as 0...')
df['target'] = df['loan_status'].map({'Bad':0, 'Good':1})

# split data into training and testing sets
print('Splitting data into training and testing sets...')
X = df.drop(['target', 'loan_status'], axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# remove multicollinear features with threshold = 0.8
remove_multicollinearity = RemoveMulticollinearity(threshold=0.8)

# remove multicollinear features with threshold = 10.0
vif_selector = VIFSelector(threshold=10.0)

# load pipeline
print('Loading pipeline...')
with open('./models/clf_pipeline.pkl','rb') as f:
    clf = pickle.load(f)

# check model score
print('Checking model score...')
print(f'Model score (training): {clf.score(X_train, y_train):.3f}') 
print(f'Model score (testing): {clf.score(X_test, y_test):.3f}')

# check auc score
print('Checking AUC score...')
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
auc_train = roc_auc_score(y_train, y_pred_train)
auc_test = roc_auc_score(y_test, y_pred_test)

# compute confusion matrix
cm = confusion_matrix(y_test, clf.predict(X_test))
tn, fp, fn, tp = cm.ravel()

# print total number of records
good = y_test[y_test == 1].count()
bad = y_test[y_test == 0].count()
tpr = round(tp / (tp+fn) * 100, 3) 
fpr = round(fp / (fp+tn) * 100, 3)

print("Total number of loan records\t:", len(y_test))
print(f"Total number of Good loans\t: {tp+fn}")
print(f"Total number of Bad loans\t: {tn+fp}")
print(f"True Positive Rate (TPR)\t: {tpr}%")
print(f"False Positive Rate (FPR)\t: {fpr}%")

print(f'AUC score (training): {auc_train:.3f}')
print(f'AUC score (testing): {auc_test:.3f}')
print('============================================')
print('============= FINISHED =====================')
print('============================================')