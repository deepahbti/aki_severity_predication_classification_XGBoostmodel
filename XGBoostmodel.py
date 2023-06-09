import json
import string
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
import csv
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, f1_score

# Read file 
df = pd.read_csv('/path/to/csv/folder')

# Changing target categorical varibale values into binary
df['POST_AKIN_CLASS'] = df['POST_AKIN_CLASS'].apply(lambda x: 1 if x > 0 else x)

# the dtype of this variable was'object' but it contains missing NUMERICAL values 
# so to impute missing values with mean need to change the dtype this variable into numeric 
df['ord_value'] = pd.to_numeric(df['ord_value'], errors='coerce')

#imputing the selected variables with '0' as the data missingness is in these variables is less than 5% 
COL_LIST = ['INTRA_COLLOID_ML',
'INTRA_XFUSION_RBC_ML',
'INTRA_XFUSION_FFP_ML',
'INTRA_XFUSION_PLT_ML',
'INTRA_CRYSTALLOID_ML','INTRA_EBL','alineid','palineid'
 ]
for col in COL_LIST:
    df[col].fillna((0), inplace=True)

# imputing the variables with mean where missingness is higher and variable are important and generating colums
# create a new column named was_imputed to track whether a value in the original column was imputed or not. 
# The was_imputed column is assigned the value 1 for the rows where the original column had null values and 0 for
# the rows where the original column had non-null values.    
COL_LIST1 = ['HCUP_CODE','anes_case_min','PRE_GLUCOSE_VAL','ord_value','paline','aline','INTRA_DUR_Map_60','age','weight_kg','bmi','GFR']
    
was_imputed = pd.DataFrame()

for column in COL_LIST1:
    mean = df[column].mean()
    df[column].fillna(mean, inplace=True)
    was_imputed[column] = df[column].isnull().astype(int)

# Add the "was_imputed" column to the original dataframe
df = pd.concat([df, was_imputed.add_suffix('_imputed')], axis=1)

#filling categorical values to 'unknown'
df.fillna('unknown', inplace=True)
for i in df:
    a= df[i].isnull().sum()
    print("{0}:{1}".format(i,a))
    
#CHANGING LABELS OF CASE_LOC
values ={'OR MSH':'MSH',
         'Anesthesia Ad Hocs MSH':'MSH',
         'Endo MSH':'MSH',
         'OR MSQ':'MSQ',
         'Endo MSQ':'MSQ',
         'Anesthesia Ad Hocs MSQ':'MSQ',
         'MSQ CEREBROVASCULAR STROKE':'MSQ',
         'MSQ INTERVENTIONAL RADIOLOGY':'MSQ',
         'Anesthesia Ad Hocs MSM ':'MSM',
         'Anesthesia Ad Hocs MSB':'MSB',
         'Anesthesia Ad Hocs MSW':'MSW',
         'MSQ Cardiology TEE':'MSQ',
         'Anesthesia Ad Hocs MSBI':'MSBI',
         'INTERVENTIONAL RADIOLOGY MSM':'MSM',}

df= df.replace({'CASE_LOC':values})
#print(df1.CASE_LOC)
    
df= df.drop('PAT_ID', axis=1)
    
#print(df.dtypes)

# perform one-hot encoding on categorical features
cat_cols = df.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(sparse=False)
encoded_cols = pd.DataFrame(encoder.fit_transform(df[cat_cols]))
encoded_cols.columns = encoder.get_feature_names_out(cat_cols)

# drop original categorical features and add encoded features
df = pd.concat([df.drop(columns=cat_cols), encoded_cols], axis=1)

# perform normalization on non-imputed and non-POST_AKIN_CLASS columns
cols_to_norm = df.columns[(~df.columns.str.contains('_imputed')) & (df.columns != 'POST_AKIN_CLASS')]
scaler = StandardScaler()
df[cols_to_norm] = scaler.fit_transform(df[cols_to_norm])

# concatenate normalized data with imputed and POST_AKIN_CLASS features
cols_to_concat = df.columns[(df.columns.str.contains('_imputed')) | (df.columns == 'POST_AKIN_CLASS')]
df = pd.concat([df[cols_to_norm], df[cols_to_concat]], axis=1)


import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, average_precision_score, roc_auc_score

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('POST_AKIN_CLASS', axis=1),
                                                    df['POST_AKIN_CLASS'], 
                                                    test_size=0.3,
                                                    random_state=42)


# Define the parameter grid
param_grid = {
    'objective': ['binary:logistic'],
    'booster': ['gbtree'],
    'eval_metric': ['auc'],
    'seed': [42],
    'verbosity': [0],
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 3, 5],
    'eta': [0.1, 0.01, 0.001],
    'gamma': [0, 0.1, 0.5],
    'subsample': [0.5, 0.7, 1],
    'colsample_bytree': [0.5, 0.7, 1],
    'nthread': [-1],
}

# Create an XGBoost classifier and a GridSearchCV object
clf = xgb.XGBClassifier()
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='roc_auc')

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)


# Get the best parameters and the model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Print the best parameter grid
print(f"Best Parameter Grid: {best_params}")

# Make predictions on the test set
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# Find the threshold that optimizes F1 score
f1_scores = []
thresholds = []
for threshold in range(0, 101):
    y_pred_th = (y_prob > threshold/100).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_th))
    thresholds.append(threshold/100)

best_threshold = thresholds[f1_scores.index(max(f1_scores))]
best_f1_score = max(f1_scores)

# Calculate evaluation metrics and confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = (tp+tn)/(tp+tn+fp+fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
specificity = tn/(tn+fp)
auprc = average_precision_score(y_test, y_prob)
auroc = roc_auc_score(y_test, y_prob)

# Print the results
print(f"Confusion matrix: \n{confusion_matrix(y_test, y_pred)}")
print(f"Accuracy:{accuracy}")
print(f"AUPRC: {auprc}")
print(f"AUROC: {auroc}")
print(f"Best threshold: {best_threshold}")
print(f"Best F1 score: {best_f1_score}")
print(f"Sensitivity: {recall}")
print(f"Specificity: {specificity}")
print(f"Threshold that optimizes F1 score: {thresholds[f1_scores.index(max(f1_scores))]}")
print(f"F1 score using the threshold: {f1_score(y_test, (y_prob > best_threshold).astype(int))}")

