# load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sklearn
import seaborn as sns
import os

os.getcwd()

# load data
data = pd.read_csv('.\\Data\\HR.csv')
data.shape
data.columns
pd.set_option('display.max_columns', None)
data.head()
data.dtypes

# clean data
data.sales.unique()
data = data.rename(columns = {'sales':'department'})
data.department.value_counts()
data.department = np.where(data.department=='support', 'technical', data.department)
data.department = np.where(data.department=='IT', 'technical', data.department)

# missing values
data.isnull().any()

# summary statistics
data.describe()

# Exploratory analysis
data.left.value_counts()
data.groupby('left').mean()
data.groupby('department').mean()
data.groupby('salary').mean()

# Plots
pd.crosstab(data.department, data.left, normalize='index').round(2)
pd.crosstab(data.department, data.left).plot(kind='bar', title='Turnover frequency by department')
pd.crosstab(data.salary, data.left, normalize='index').plot(kind='bar', stacked=True, title='Turnover frequency by salary')

# histograms
data.hist(bins=10, figsize=(15,10))

# Create dummy vars for categorical vars
cat_vars = ['department', 'salary']
for var in cat_vars:
    cat_dummy = pd.get_dummies(data[var], prefix=var)
    data = data.join(cat_dummy)
data = data.drop(['department', 'salary'], axis=1)

# Assign independent & outcome vars
ind_vars = data.columns.values.tolist()
X = [i for i in ind_vars if i not in 'left']

# Feature selection
from sklearn.feature_selection import RFE # identify features which contribute most to predicting target attribute
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=15000)
rfe = RFE(model, 10)
rfe = rfe.fit(data[X], data['left'])
rfe.support_
rfe.ranking_

features = data[X].columns[np.where(rfe.ranking_==1)]

X = data[features]
y = data['left']


# Logistic Regression Model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

from sklearn import metrics
from sklearn.metrics import accuracy_score
accuracy_score(y_test, logreg.predict(X_test))

# Random Forest Model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

accuracy_score(y_test, rf.predict(X_test))

# Support Vector Machine
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)

accuracy_score(y_test, svc.predict(X_test))

print('Logistic Regression accuracy: {:.3f}'.format(accuracy_score(y_test, logreg.predict(X_test))))
print('Random Forest accuracy: {:.3f}'.format(accuracy_score(y_test, rf.predict(X_test))))
print('SVM accuracy: {:.3f}'.format(accuracy_score(y_test, svc.predict(X_test))))

# Cross Validation on RF model
# 10-fold cross-validation train RF model
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10)
modelCV = RandomForestClassifier()

scores = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring='accuracy')
print('10-fold cross validation average accuracy: %.3f' % (scores.mean())) 
# Mean accuracy close to RF model accuracy - model generalizes well

# Precision & recall
from sklearn.metrics import classification_report
print(classification_report(y_test, logreg.predict(X_test)))
print(classification_report(y_test, rf.predict(X_test)))

from sklearn.metrics import confusion_matrix
logreg_y_pred = logreg.predict(X_test)
logreg_cm = metrics.confusion_matrix(logreg_y_pred, y_test, [1,0])
sns.heatmap(logreg_cm, annot=True, fmt='.2f', xticklabels=['Left','Stay'], yticklabels=['Left','Stay'])
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Logistic Regression model')

rf_y_pred = rf.predict(X_test)
rf_cm = metrics.confusion_matrix(rf_y_pred, y_test, [1,0])
sns.heatmap(rf_cm, annot=True, fmt='.2f', xticklabels=['Left','Stay'], yticklabels=['Left','Stay'])
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Random Forest model')


