
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.pop('EmployeeNumber')
df.pop('Over18')
df.pop('StandardHours')
df.pop('EmployeeCount')
y = df['Attrition']
tmp = df['Attrition']
X = df
X.pop('Attrition')
y.unique()
le= preprocessing.LabelBinarizer()
y= le.fit_transform(y)
tmp = le.fit_transform(tmp)
tmp = pd.Series(list(tmp))
ind_BusinessTravel = pd.get_dummies(df['BusinessTravel'])
ind_Department = pd.get_dummies(df['Department'])
ind_EducationField = pd.get_dummies(df['EducationField'])
ind_Gender = pd.get_dummies(df['Gender'])
ind_JobRole = pd.get_dummies(df['JobRole'])
ind_MaritalStatus = pd.get_dummies(df['MaritalStatus'])
ind_OverTime = pd.get_dummies(df['OverTime'])
df['BusinessTravel'].unique()
#df1 = pd.concat([ind_BusinessTravel, ind_Department, ind_EducationField, ind_Gender, 
               #  ind_JobRole, ind_MaritalStatus, ind_OverTime])
df1 = pd.concat([ind_BusinessTravel, ind_Department, ind_EducationField, ind_Gender, 
                 ind_JobRole, ind_MaritalStatus, ind_OverTime, df.select_dtypes(['int64'])], axis=1)

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    '''
    print the accuracy score, classification report and confusion matrix of classifier
    '''
    if train:
        '''
        training performance
        '''
        print("Train Result:\n")
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, clf.predict(X_train))))
        print("Classification Report: \n {}\n".format(classification_report(y_train, clf.predict(X_train))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, clf.predict(X_train))))

        res = cross_val_score(clf, X_train, y_train.ravel(), cv=10, scoring='accuracy')
        print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))
        
    elif train==False:
        '''
        test performance
        '''
        print("Test Result:\n")        
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, clf.predict(X_test))))
        print("Classification Report: \n {}\n".format(classification_report(y_test, clf.predict(X_test))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, clf.predict(X_test))))    

X_train, X_test, y_train, y_test = train_test_split(df1, y)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
bag_clf = BaggingClassifier(base_estimator= clf, n_estimators=5000,bootstrap=True, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train.ravel())
print("\n\n-----Bagging-----\n\n")
print_score(bag_clf, X_train, y_train, X_test, y_test, train=True)
print_score(bag_clf, X_train, y_train, X_test, y_test, train=False)
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train.ravel())
print("\n\n-----Random Forest-----\n\n")
print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)
pd.Series(rf_clf.feature_importances_, index=X_train.columns).sort_values(ascending=False).plot(kind='bar', figsize=(12,6));
ada_clf = AdaBoostClassifier()
ada_clf.fit(X_train, y_train.ravel())
print("\n\n-----AdaBoost-----\n\n")
print_score(ada_clf, X_train, y_train, X_test, y_test, train=True)
print_score(ada_clf, X_train, y_train, X_test, y_test, train=False)
ada_clf = AdaBoostClassifier(RandomForestClassifier())
ada_clf.fit(X_train, y_train.ravel())
print("\n\n-----AdaBoost + RandomForest-----\n\n")
print_score(ada_clf, X_train, y_train, X_test, y_test, train=True)
print_score(ada_clf, X_train, y_train, X_test, y_test, train=False)
gbc_clf = GradientBoostingClassifier()
gbc_clf.fit(X_train, y_train.ravel())
print("\n\n-----Gradient Boosting-----\n\n")
print_score(gbc_clf, X_train, y_train, X_test, y_test, train=True)
print_score(gbc_clf, X_train, y_train, X_test, y_test, train=False)
xgb_clf = xgb.XGBClassifier()
xgb_clf.fit(X_train, y_train.ravel())
print("\n\n-----XGBoost-----\n\n")
print_score(xgb_clf, X_train, y_train, X_test, y_test, train=True)
print_score(xgb_clf, X_train, y_train, X_test, y_test, train=False)

