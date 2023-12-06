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
from sklearn.model_selection import cross_val_score

# Helper function
from matrix_metric.py import confusion_matrix_metrics

# Importing data

df = np.loadtxt("insert file path",
                 delimiter=",", skiprows = 1,).astype(np.int64)

# Preprocessing and Feature Engineering

# Capturing the features and labels numerical values
features = df[1:, 1:49].clip(min=0) #setting negative values to 0
labels = df[1:, -1:]


# Running features through decision tree to determine relevant features
decisionModel = DecisionTreeClassifier()
decisionModel = decisionModel.fit(features, labels)
importance = decisionModel.feature_importances_

# List of features and variances
for i, v in enumerate(importance):
  print('Feature: %0d, Score: %.5f' % (i,v))



plt.bar([x for x in range(len(importance))], importance)
plt.title("Naive Bayes Feature Importance")
plt.xlabel("Features")
plt.ylabel("Score")
plt.show()

# Feature Selection
print(features.shape)
model = SelectFromModel(decisionModel, prefit=True)
NX_all = model.transform(features)
print(NX_all.shape)

# NAIVE BAYES TRAINING
# Model training

# splitting dataset
NX_train, NX_test, NY_train, NY_test = train_test_split(NX_all, labels, test_size = 0.2, random_state = 42)


# training the model
nbModel = MultinomialNB(force_alpha=True).fit(NX_train, NY_train.ravel())
NY_pred = nbModel.predict(NX_test)

# construct confusion matrix
tn, fp, fn, tp = confusion_matrix(NY_test,NY_pred).ravel()

# matrix metrics
confusion_matrix_metrics(tp, fp, tn, fn)

# ROC curve calculation
fpr, tpr, thresholds = roc_curve(NY_test, NY_pred)
roc_auc = roc_auc_score(NY_test, NY_pred)

# Display Plots

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Confusion Matrix subplot
cmatrix = ConfusionMatrixDisplay.from_predictions(NY_test, NY_pred, ax=ax1)
ax1.set_title('Naive Bayes Confusion Matrix')

# ROC Curve subplot
ax2.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax2.plot([0, 1], [0, 1], 'k--')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('Naive Bayes ROC Curve')
ax2.legend()

plt.show()

# BOOTSTRAPPING
n_bootstrap = 1050

bootstrap_stats = []

# Perform bootstrapping

for _ in range(n_bootstrap):
  sample = np.random.choice(len(features), size=len(features), replace=True)
  bootstrap_features = features[sample]
  bootstrap_labels = labels[sample]
  bootstrap_stats.append(bootstrap_features.mean(axis=0))

# running features through decision tree to determine relevant features
decisionModel = DecisionTreeClassifier() #bootstrap depth = 13
decisionModel = decisionModel.fit(bootstrap_features, bootstrap_labels)
importance = decisionModel.feature_importances_

# list of features and variances
for i, v in enumerate(importance):
  print('Feature: %0d, Score: %.5f' % (i,v))



plt.bar([x for x in range(len(importance))], importance)
plt.title("Bootstrap Feature Importance")
plt.xlabel("Features")
plt.ylabel("Score")
plt.show()

print(bootstrap_features.shape)
model = SelectFromModel(decisionModel, prefit=True)
BX_all = model.transform(bootstrap_features)
print(BX_all.shape)

# BOOTSTRAP TRAINING
# retraining model
BX_train, BX_test, BY_train, BY_test = train_test_split(bootstrap_features, bootstrap_labels, test_size = 0.2, random_state = 42)

# training the model
nbModel = MultinomialNB(force_alpha=True).fit(BX_train, BY_train.ravel())
BY_pred = nbModel.predict(BX_test)

# construct and display confusion matrix
tn, fp, fn, tp = confusion_matrix(BY_test,BY_pred).ravel()

# matrix metrics
confusion_matrix_metrics(tp, fp, tn, fn)

# ROC curve calculation
fpr, tpr, thresholds = roc_curve(BY_test, BY_pred)
roc_auc = roc_auc_score(BY_test, BY_pred)

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Confusion Matrix subplot
cmatrix = ConfusionMatrixDisplay.from_predictions(BY_test, BY_pred, ax=ax1)
ax1.set_title('Bootstrap Confusion Matrix')
fpr, tpr, thresholds = roc_curve(BY_test, BY_pred)
roc_auc = roc_auc_score(BY_test, BY_pred)
# ROC Curve subplot
ax2.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax2.plot([0, 1], [0, 1], 'k--')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('Naive Bayes ROC Curve')
ax2.legend()

plt.show()

# BAGGING
num_models = 1050
model_preds = []

for _ in range(num_models):
  BGX_train, BGX_test, BGY_train, BGY_test = train_test_split(bootstrap_features, bootstrap_labels, test_size = 0.2, random_state = 42, stratify=labels)
  nbModel = MultinomialNB(force_alpha=True).fit(BGX_train, BGY_train.ravel())
  model_preds.append(nbModel.predict(BGX_test))


ensemble_predictions = np.mean(model_preds, axis=0)  # You can also use np.median() for voting

# Evaluate the ensemble model's metrics
ensemble_accuracy = np.mean(ensemble_predictions == BGY_test)


tn, fp, fn, tp = confusion_matrix(BGY_test, ensemble_predictions).ravel()
print("Ensemble Model Accuracy:", ensemble_accuracy)
confusion_matrix_metrics(tp, fp, tn, fn,True)

# Ada Boosting
AX_train, AX_test, AY_train, AY_test = train_test_split(NX_all, labels, test_size = 0.2, random_state = 42)

base_model = DecisionTreeClassifier(max_depth=1)

adaboost_classifier = AdaBoostClassifier(base_model, n_estimators=50, random_state=42).fit(AX_train, AY_train)

AY_pred = adaboost_classifier.predict(AX_test)

# construct and display confusion matrix
tn, fp, fn, tp = confusion_matrix(AY_test,AY_pred).ravel()

# matrix metrics
confusion_matrix_metrics(tp, fp, tn, fn)

# ROC curve calculation
fpr, tpr, thresholds = roc_curve(AY_test, AY_pred)
roc_auc = roc_auc_score(AY_test, AY_pred)

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Confusion Matrix subplot
cmatrix = ConfusionMatrixDisplay.from_predictions(AY_test, AY_pred, ax=ax1)
ax1.set_title('Ada Confusion Matrix')
fpr, tpr, thresholds = roc_curve(AY_test, AY_pred)
roc_auc = roc_auc_score(AY_test, AY_pred)
# ROC Curve subplot
ax2.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax2.plot([0, 1], [0, 1], 'k--')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('Naive Bayes ROC Curve')
ax2.legend()

plt.show()

# CROSS-VALIDATION
# Naive Bayes
train_scores, test_scores = list(), list()
# define the tree depths to evaluate
values = [i for i in range(1, 21)]


# evaluate a decision tree for each depth
for i in values:
 # configure the model
 model = DecisionTreeClassifier(max_depth=i)
 # fit model on the training dataset
 model.fit(NX_train, NY_train)
 # evaluate on the train dataset
 train_yhat = model.predict(NX_train)
 train_acc = accuracy_score(NY_train, train_yhat)
 train_scores.append(train_acc)
 # evaluate on the test dataset
 test_yhat = model.predict(NX_test)
 test_acc = accuracy_score(NY_test, test_yhat)
 test_scores.append(test_acc)
 # summarize progress
 print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))

# K-Folds cross validation
# define the tree depths to evaluate
values = [i for i in range(1, 21)]

depths = []
mean_scores = []
# evaluate a decision tree for each depth
for i in values:
 # configure the model
 model = DecisionTreeClassifier(max_depth=i)
 # Perform 7-fold cross validation
 scores = cross_val_score(estimator=model, X=features, y=labels, cv=7, n_jobs=4)
 mean_scores.append(scores.mean())
 depths.append((i,scores.mean()))

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


# plot of train and test scores vs tree depth
ax1.plot(values, mean_scores, '-o')
ax1.set_title('Naive Bayes Cross Validation')
ax1.set_xlabel("Mean Values")
ax1.set_ylabel("Scores")

# plot of train and test scores vs tree depth
ax2.plot(values, train_scores, '-o', label='Train')
ax2.plot(values, test_scores, '-o', label='Test')
ax2.set_title('Naive Bayes Cross Validation')
ax2.set_xlabel("Values")
ax2.set_ylabel("Scores")
ax2.legend()


plt.show()


# Bootstrap
train_scores, test_scores = list(), list()

# define the tree depths to evaluate
values = [i for i in range(1, 21)]


# evaluate a decision tree for each depth
for i in values:
 # configure the model
 model = DecisionTreeClassifier(max_depth=i)
 # fit model on the training dataset
 model.fit(BX_train, BY_train)
 # evaluate on the train dataset
 train_yhat = model.predict(BX_train)
 train_acc = accuracy_score(BY_train, train_yhat)
 train_scores.append(train_acc)
 # evaluate on the test dataset
 test_yhat = model.predict(BX_test)
 test_acc = accuracy_score(BY_test, test_yhat)
 test_scores.append(test_acc)
 # summarize progress
 print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))

# K folds booststrap
# define the tree depths to evaluate
values = [i for i in range(1, 21)]

depths = []
mean_scores = []
# evaluate a decision tree for each depth
for i in values:
 # configure the model
 model = DecisionTreeClassifier(max_depth=i)
 # Perform 7-fold cross validation
 scores = cross_val_score(estimator=model, X=bootstrap_features, y=bootstrap_labels, cv=7, n_jobs=4)
 mean_scores.append(scores.mean())
 depths.append((i,scores.mean()))

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


# plot of train and test scores vs tree depth
ax1.plot(values, mean_scores, '-o')
ax1.set_title('Bootstrap Cross Validation')
ax1.set_xlabel("Mean Values")
ax1.set_ylabel("Scores")



# plot of train and test scores vs tree depth
ax2.plot(values, train_scores, '-o', label='Train')
ax2.plot(values, test_scores, '-o', label='Test')
ax2.set_title('Bootstrap Cross Validation')
ax2.set_xlabel("Values")
ax2.set_ylabel("Scores")
ax2.legend()


plt.show()


# CROSS VALIDATED NAIVE BAYES
# Feature Engineering

# running features through decision tree to determine relevant features with optimal tree depth
decisionModel = DecisionTreeClassifier(max_depth=10) 
decisionModel = decisionModel.fit(features, labels)
importance = decisionModel.feature_importances_

# list of features and variances
for i, v in enumerate(importance):
  print('Feature: %0d, Score: %.5f' % (i,v))



plt.bar([x for x in range(len(importance))], importance)
plt.title("Cross Validation Naives Bayes Feature Importance")
plt.xlabel("Features")
plt.ylabel("Score")
plt.show()

# Feature Selection
print(features.shape)
model = SelectFromModel(decisionModel, prefit=True)
NX_all = model.transform(features)
print(NX_all.shape)

# CROSS VALIDATED NAIVE BAYES TRAINING AND PLOT
# splitting dataset
NX_train, NX_test, NY_train, NY_test = train_test_split(NX_all, labels, test_size = 0.2, random_state = 42)

# training the model
nbModel = MultinomialNB(force_alpha=True).fit(NX_train, NY_train.ravel())
NY_pred = nbModel.predict(NX_test)

# construct and display confusion matrix
tn, fp, fn, tp = confusion_matrix(NY_test,NY_pred).ravel()

# matrix metrics
confusion_matrix_metrics(tp, fp, tn, fn)

# ROC curve calculation
fpr, tpr, thresholds = roc_curve(NY_test, NY_pred)
roc_auc = roc_auc_score(NY_test, NY_pred)

# Display Plots

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Confusion Matrix subplot
cmatrix = ConfusionMatrixDisplay.from_predictions(NY_test, NY_pred, ax=ax1)
ax1.set_title('Cross Validated Naive Bayes Confusion Matrix')

# ROC Curve subplot
ax2.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax2.plot([0, 1], [0, 1], 'k--')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('Cross Validated Naive Bayes ROC Curve')
ax2.legend()

plt.show()

# CROSS VALIDATED BOOTSTRAP

# running features through decision tree to determine relevant features with optimal tree depth
decisionModel = DecisionTreeClassifier(max_depth=13)
decisionModel = decisionModel.fit(bootstrap_features, bootstrap_labels)
importance = decisionModel.feature_importances_

# list of features and variances
for i, v in enumerate(importance):
  print('Feature: %0d, Score: %.5f' % (i,v))


plt.bar([x for x in range(len(importance))], importance)
plt.title("Cross Validation Bootstrap Feature Importance")
plt.xlabel("Features")
plt.ylabel("Score")
plt.show()

print(bootstrap_features.shape)
model = SelectFromModel(decisionModel, prefit=True)
BX_all = model.transform(bootstrap_features)
print(BX_all.shape)


# CROSS VALIDATED BOOTSTRAP TRAINING
# retraining model
BX_train, BX_test, BY_train, BY_test = train_test_split(bootstrap_features, bootstrap_labels, test_size = 0.2, random_state = 42)

# training the model
nbModel = MultinomialNB(force_alpha=True).fit(BX_train, BY_train.ravel())
BY_pred = nbModel.predict(BX_test)

# construct and display confusion matrix
tn, fp, fn, tp = confusion_matrix(BY_test,BY_pred).ravel()

# matrix metrics
confusion_matrix_metrics(tp, fp, tn, fn)

# ROC curve calculation
fpr, tpr, thresholds = roc_curve(BY_test, BY_pred)
roc_auc = roc_auc_score(BY_test, BY_pred)

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Confusion Matrix subplot
cmatrix = ConfusionMatrixDisplay.from_predictions(BY_test, BY_pred, ax=ax1)
ax1.set_title(' CV Bootstrap Confusion Matrix')
fpr, tpr, thresholds = roc_curve(BY_test, BY_pred)
roc_auc = roc_auc_score(BY_test, BY_pred)
# ROC Curve subplot
ax2.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax2.plot([0, 1], [0, 1], 'k--')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('CV Bootstrap ROC Curve')
ax2.legend()

plt.show()
