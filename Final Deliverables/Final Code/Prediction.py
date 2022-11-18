

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


df=pd.read_csv('Heart_Disease_Prediction.csv')

#Data Visualization

count=1
plt.subplots(figsize=(30,25))
for i in df.columns:
    if df[i].dtypes!='object':
        plt.subplot(6,7,count)
        sns.distplot(df[i])
        count+=1

plt.show()

count=1
plt.subplots(figsize=(30,25))
for i in df.columns:
    if df[i].dtypes!='object':
        plt.subplot(6,7,count)
        sns.boxplot(df[i])
        count+=1

plt.show()



from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df.iloc[:,-1]=le.fit_transform(df.iloc[:,-1])

"""# Train test and split """

X = df.iloc[:,:-1]
Y = df.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)



"""# Accuracies of different algorithms applied"""


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from math import sqrt
# %matplotlib inline

from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2,random_state =42)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

y_pred_lr = logreg.predict(X_test)

score_lr = round(accuracy_score(y_pred_lr,Y_test)*100,2)
print(score_lr)

print(classification_report(Y_test, y_pred_lr))

from sklearn.metrics import confusion_matrix
matrix= confusion_matrix(Y_test, y_pred_lr)
sns.heatmap(matrix,annot = True, fmt = "d")


#Random Forest
from sklearn.ensemble import RandomForestClassifier
randfor = RandomForestClassifier(n_estimators=100, random_state=0)

randfor.fit(X_train, Y_train)

y_pred_rf = randfor.predict(X_test)

score_rf = round(accuracy_score(y_pred_rf,Y_test)*100,2)
print(score_rf)

print(classification_report(Y_test, y_pred_rf))

matrix= confusion_matrix(Y_test, y_pred_rf)
sns.heatmap(matrix,annot = True, fmt = "d")

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=3, random_state=0)

dt.fit(X_train, Y_train)

y_pred_dt = dt.predict(X_test)

score_dt = round(accuracy_score(y_pred_dt,Y_test)*100,2)

print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")

print(classification_report(Y_test, y_pred_dt))

matrix= confusion_matrix(Y_test, y_pred_dt)
sns.heatmap(matrix,annot = True, fmt = "d")

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred_nb = gnb.fit(X_train, Y_train).predict(X_test)

score_nb = round(accuracy_score(y_pred_nb,Y_test)*100,2)

print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")

print(classification_report(Y_test, y_pred_nb))

matrix= confusion_matrix(Y_test, y_pred_nb)
sns.heatmap(matrix,annot = True, fmt = "d")

#SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)
y_pred_svm = dt.predict(X_test)

score_svm = round(accuracy_score(y_pred_svm,Y_test)*100,2)

print("The accuracy score achieved using SVM is: "+str(score_svm)+" %")

print(classification_report(Y_test, y_pred_svm))

matrix= confusion_matrix(Y_test, y_pred_svm)
sns.heatmap(matrix,annot = True, fmt = "d")

accuracy = []

# list of algorithms names
classifiers = ['Decision Trees', 'Logistic Regression', 'Naive Bayes', 'Random Forests','Support Vector Machine']

# list of algorithms with parameters
models = [DecisionTreeClassifier(max_depth=3, random_state=0), LogisticRegression(), 
        GaussianNB(), RandomForestClassifier(n_estimators=100, random_state=0), SVC(kernel = 'rbf', random_state = 0)]

# loop through algorithms and append the score into the list
for i in models:
    model = i
    model.fit(X_train, Y_train)
    score = model.score(X_test, Y_test)
    accuracy.append(score)

summary = pd.DataFrame({'accuracy':accuracy}, index=classifiers)       
summary



"""# Saving the Model

Logistic Regression and Naive Bayes have the highest accuracy compared to other algorithms.
So we can save either one of them.
"""



import pickle
filename = 'finalized_model.sav'
pickle.dump(logreg, open(filename, 'wb'))
 



