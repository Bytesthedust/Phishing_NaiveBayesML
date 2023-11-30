# Author: Kevon Nelson

# Dependencies
from urllib.request import urlopen # A library for working with URLs.
import numpy as np # A library for numerical operations and array manipulations.
from matplotlib import pyplot as plt # A library for creating visualizations, used here for plotting.

#- scikit-learn: A machine learning library used for metrics, Naive Bayes models, and visualization.
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.ensemble import AdaBoostClassifier


#Importing data

df = np.loadtxt("/content/Phishing_Legitimate_full.csv",
                 delimiter=",", skiprows = 1,).astype(np.int64)

#Preprocessing and Feature Engineering

#capturing the features and labels numerical values
features = df[1:, 1:49].clip(min=0) #setting negative values to 0
labels = df[1:, -1:]


#running features through decision tree to determine relevant features
decisionModel = DecisionTreeClassifier() #max depth = 10
decisionModel = decisionModel.fit(features, labels)
importance = decisionModel.feature_importances_

#list of features and variances
for i, v in enumerate(importance):
  print('Feature: %0d, Score: %.5f' % (i,v))



plt.bar([x for x in range(len(importance))], importance)
plt.title("Naive Bayes Feature Importance")
plt.xlabel("Features")
plt.ylabel("Score")
plt.show()

#Feature Selection
print(features.shape)
model = SelectFromModel(decisionModel, prefit=True)
NX_all = model.transform(features)
print(NX_all.shape)
