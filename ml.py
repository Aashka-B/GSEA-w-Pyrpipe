import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import ttest_ind
import os
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from sanbomics.plots import volcano
import gseapy as gp
from gseapy import barplot, dotplot
from gseapy.plot import gseaplot
from sanbomics.tools import id_map
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import xgboost as xgb
from skopt import BayesSearchCV

# Import data
sigml = pd.read_csv(r"/courses/BINF6310.202410/students/bhowmick.a/PanC/ml.txt", sep = '\t')
sigml= sigml.drop(columns= ['Unnamed: 0'])

# Use smote to reduce bias
smote = SMOTE()

# Defining X, y, for training ML models
X = sigml.iloc[:,:-1]
y = sigml['state']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=52)

features = list(X.columns)

X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Logistic regression
lrmodel= LogisticRegression(C=10.0, class_weight='balanced', max_iter=10000, multi_class='ovr')
lrmodel.fit(X_resampled, y_resampled)

y_pred = lrmodel.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(cm)

report = classification_report(y_test,y_pred) # for a report regarding the evaluation
print(report)

lrcoefs = lrmodel.coef_ # take coefficients list
sortedlrcoefsarr = sorted(lrcoefs, key= abs) # sort the coefficients
sortedlrcoefs = [x for y in sortedlrcoefsarr for x in y]

# Figures LR
fpr, tpr, thresholds = roc_curve(y_test, y_pred) # to retrieve false positive and true
auc = metrics.auc(fpr,tpr) # calculate AUC
precision, recall, thresholds = precision_recall_curve(y_test, y_pred) # calculate PR
# Create a new figure with two subplots side by side
plt.figure(figsize=(12, 4))
# First subplot for ROC curve
plt.subplot(121) # 1 row, 2 columns, first subplot
plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('A. ROC curve for logistic regression model')
plt.legend()

# Save the figure to the specified directory and filename
save_path = "/courses/BINF6310.202410/students/bhowmick.a/PanC/ML/LOC_curve_for_LRM.png"
plt.savefig(save_path)
plt.close()  # Close the figure to free up resources

# Second subplot for PR curve
plt.subplot(122) # 1 row, 2 columns, second subplot
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('B. Logistic Regression PR Curve')
plt.tight_layout() # So that the subplots don't overlap

# Save the figure to the specified directory and filename
save_path = "/courses/BINF6310.202410/students/bhowmick.a/PanC/ML/LR_PR_curve.png"
plt.savefig(save_path)
plt.close()  # Close the figure to free up resources

plt.figure(figsize = (12,6))
plt.bar(features,sortedlrcoefs)
plt.xticks(rotation = 90)
plt.xlabel("Feature")
plt.ylabel("Coefficient")
plt.title("Coefficients of Logistic Regression Model")
plt.tight_layout()

# Save the figure to the specified directory and filename
save_path = "/courses/BINF6310.202410/students/bhowmick.a/PanC/ML/Coefficients_of_LRM.png"
plt.savefig(save_path)
plt.close()  # Close the figure to free up resources

svmbest= SVC(C=100,class_weight= 'balanced', coef0=0.05, degree=2, gamma='auto', kernel='poly')
svmbest.fit(X_resampled,y_resampled)

y_pred = svmbest.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(cm)

report = classification_report(y_test,y_pred)
print(report)

# SVM Figures
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc = metrics.auc(fpr,tpr)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
plt.figure(figsize=(12, 4))
# First subplot for ROC curve
plt.subplot(121) # 1 row, 2 columns, first subplot
plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('A. ROC curve for support vector machine')
plt.legend()

# Save the figure to the specified directory and filename
save_path = "/courses/BINF6310.202410/students/bhowmick.a/PanC/ML/ROC_curve_for_SVM.png"
plt.savefig(save_path)
plt.close()  # Close the figure to free up resources

# Second subplot for PR curve
plt.subplot(122) # 1 row, 2 columns, second subplot
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('B. Support vector machine PR Curve')

plt.tight_layout() # So that the subplots don't overlap

# Save the figure to the specified directory and filename
save_path = "/courses/BINF6310.202410/students/bhowmick.a/PanC/ML/SVM_PR_curve.png"
plt.savefig(save_path)
plt.close()  # Close the figure to free up resources

# Random Forest Classifier
bestfc = RandomForestClassifier(ccp_alpha=0.0,
class_weight= 'balanced',max_depth=9,
max_features='log2',
max_leaf_nodes=18,
max_samples=16,
min_impurity_decrease= 0.05,
min_samples_split=3,
min_samples_leaf=1,
min_weight_fraction_leaf=0.07,
n_estimators=493,
criterion = 'entropy')

bestfc.fit(X_resampled,y_resampled)

y_pred= bestfc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(cm)

report = classification_report(y_test,y_pred)
print(report)

# Figures
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc = metrics.auc(fpr,tpr)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
plt.figure(figsize=(12, 4))
# previously defined AUC and fpr/tpr for single figures
# First subplot for ROC curve
plt.subplot(121) # 1 row, 2 columns, first subplot
plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('A. ROC curve for random forest classifier')
plt.legend()

# Save the figure to the specified directory and filename
save_path = "/courses/BINF6310.202410/students/bhowmick.a/PanC/ML/ROC_curve_for_RFC.png"
plt.savefig(save_path)
plt.close()  # Close the figure to free up resources

# Second subplot for PR curve
plt.subplot(122) # 1 row, 2 columns, second subplot
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('B. Random forest classifier PR Curve')

plt.tight_layout() # So that the subplots don't overlap

# Save the figure to the specified directory and filename
save_path = "/courses/BINF6310.202410/students/bhowmick.a/PanC/ML/RFC_PR_curve.png"
plt.savefig(save_path)
plt.close()  # Close the figure to free up resources

# XGBoost Clasifier
goodx = xgb.XGBClassifier(base_score= 0.16,
booster= 'dart',
colsample_bylevel= 0.2,
colsample_bynode=0.2,
colsample_bytree= 0.2,
gamma= 0.15,
grow_policy= 'lossguide',
importance_type= 'cover',
learning_rate= 0.0925,
max_bin=38,
max_delta_step= 12.05,
max_depth=10,
max_leaves=21,
min_child_weight= 0.2,
n_estimators= 230,
num_parallel_tree= 7,
objective= 'binary:logistic',
reg_lambda= 0.02,
subsample=0.85)

goodx.fit(X_resampled,y_resampled)
y_pred = goodx.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(cm)

report = classification_report(y_test,y_pred)
print(report)

# Figures
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc = metrics.auc(fpr,tpr)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
plt.figure(figsize=(12, 4))
# previously defined AUC and fpr/tpr for single figures
# First subplot for ROC curve
plt.subplot(121) # 1 row, 2 columns, first subplot
plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('A. ROC curve for XGBoost')
plt.legend()

# Save the figure to the specified directory and filename
save_path = "/courses/BINF6310.202410/students/bhowmick.a/PanC/ML/ROC_curve_for_XGBoost.png"
plt.savefig(save_path)
plt.close()  # Close the figure to free up resources

# Second subplot for PR curve
plt.subplot(122) # 1 row, 2 columns, second subplot
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('B. XGBoost PR Curve')

plt.tight_layout() # So that the subplots don't overlap

# Save the figure to the specified directory and filename
save_path = "/courses/BINF6310.202410/students/bhowmick.a/PanC/ML/XGBoost_PR_curve.png"
plt.savefig(save_path)
plt.close()  # Close the figure to free up resources
