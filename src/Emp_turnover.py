import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from pylab import savefig
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


hr = pd.read_csv('HR_dataset.csv')
col_names = hr.columns.tolist()
print("Column names:")
print(col_names)
print("\nSample data:")



hr=hr.rename(columns = {'sales':'department'})

#Data Visualization

#Bar chart for department employee work for and the frequency of turnover
pd.crosstab(hr.department,hr.left).plot(kind='bar')
plt.title('Turnover Frequency for Department')
plt.xlabel('Department')
plt.ylabel('Frequency of Turnover')
plt.savefig('department_bar_chart')

# Bar chart for employee salary level and the frequency of turnover

table=pd.crosstab(hr.salary, hr.left)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Salary Level vs Turnover')
plt.xlabel('Salary Level')
plt.ylabel('Proportion of Employees')
plt.savefig('salary_bar_chart')


#Data Preprocessing

# The “left” column is the outcome variable recording one and 0. 1 for employees who left the company and 0 for those who didn’t.

print(hr['department'].unique())


#Reduceing the categories for better modelling
hr['department']=np.where(hr['department'] =='support', 'technical', hr['department'])
hr['department']=np.where(hr['department'] =='IT', 'technical', hr['department'])

#Creating Variables for Categorical Variables

cat_vars=['department','salary']
for var in cat_vars:
    cat_list ='var'+'_'+var
    cat_list = pd.get_dummies(hr[var], prefix=var)
    hr1=hr.join(cat_list)
    hr=hr1

hr.drop(hr.columns[[8, 9]], axis=1, inplace=True)
#print(hr.columns.values)

#Correlation Matrix

corr = hr.corr()
corr = (corr)
plotcorr = sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

figure = plotcorr.get_figure()
figure.savefig('plot_corr.png',dpi=400)
print(corr)

#The outcome variable is “left”, and all the other variables are predictors.
hr_vars=hr.columns.values.tolist()
y=['left']
X=[i for i in hr_vars if i not in y]


#Feature Selection
model = LogisticRegression()
rfe = RFE(model, n_features_to_select=10)  # using The Recursive Feature Elimination (RFE) to select features

#RFE choose the 10 variables for us, which are marked True in the support_ array and marked with a choice “1” in the ranking_array.


rfe = rfe.fit(hr[X], hr[y].values.ravel())  #.values will give the values in a numpy array (shape: (n,1)) .ravel will convert that array shape to (n, ) (i.e. flatten it)


cols=['satisfaction_level', 'last_evaluation', 'time_spend_company', 'Work_accident', 'promotion_last_5years',
      'department_RandD', 'department_hr', 'department_management', 'salary_high', 'salary_low']
X=hr[cols]
y=hr['left']

#Train and Test Set to Predict Employee Turnover
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


#Using Logistics Regression Classification model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

#Accuracy of LR
print('Logistic regression accuracy: {:.3f}'.format(accuracy_score(y_test, logreg.predict(X_test))))

#Using Random Forest Classification model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
random_forest_predic = rf.predict(X_test)

# Accuracy of RF classifier
print('Random Forest Accuracy: {:.3f}'.format(accuracy_score(y_test, rf.predict(X_test))))


#Classification report for models

print("RF model\n" + classification_report(y_test, rf.predict(X_test)))
print("LG model\n" + classification_report(y_test, logreg.predict(X_test)))

#Confusion matrix/ Prediction from random forest classifier
y_pred = rf.predict(X_test)
forest_cm = metrics.confusion_matrix(y_pred, y_test)
sns.heatmap(forest_cm, annot=True, fmt='.2f',xticklabels = ["Left", "Stayed"] , yticklabels = ["Left", "Stayed"] )
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Random Forest')
plt.savefig('random_forest')

##Confusion matrix/ Prediction from LR classifier
logreg_y_pred = logreg.predict(X_test)
logreg_cm = metrics.confusion_matrix(logreg_y_pred, y_test)
sns.heatmap(logreg_cm, annot=True, fmt='.2f',xticklabels = ["Left", "Stayed"] , yticklabels = ["Left", "Stayed"] )
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Logistic Regression')
plt.savefig('logistic_regression')


#Let’s have a look at the feature importance of our random forest classification model.

feature_labels = np.array(['satisfaction_level', 'last_evaluation', 'time_spend_company', 'Work_accident', 'promotion_last_5years',
      'department_RandD', 'department_hr', 'department_management', 'salary_high', 'salary_low'])
importance = rf.feature_importances_
feature_indexes_by_importance = importance.argsort()
print("Feature importance for our RF classifier")
for index in feature_indexes_by_importance:
    print('{}-{:.2f}%'.format(feature_labels[index], (importance[index] *100.0)))


#ROC curve of
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
rf_roc_auc = roc_auc_score(y_test, rf.predict(X_test))
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic(ROC)')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png', bbox_inches='tight')