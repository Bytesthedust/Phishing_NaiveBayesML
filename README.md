# Phishing_NaiveBayesML
For CPTS_437 - Intro to Machine Learning. A Naive Bayes machine learning project to train a model to detect phishing emails

**Project Statement**
The project goal is to train a classification model to accurately predict if a given email is a phishing attack.

Phishing is a type of social engineering attack designed to deceive the target into divulging sensitive information or providing access to a network. The type of Phishing this model will be trained on is a Business Email Compromise attack where a threat actor(hacker) sends an email message that seems to be from a known source to make a seemingly legitimate request for information, in order to obtain a financial advantage.

**Algorithms**
In classifying the emails, the Naive Bayes algorithm was chosen given its regular use for detecting spam email with high accuracy. For feature engineering, the Decision Tree algorithm was chosen.

**Data**
The source of the data comes from the “Phishing Dataset for Machine Learning” Kaggle dataset which contains 48 features extracted from 10000 webpages, half of which are legitimate. The features are the frequencies of certain characters in a URL.

**Visualization**
Feature engineering will be plotted on a bar graph showing each feature and the amount of relevance they have on the model. A confusion matrix and a ROC curve will be plotted to visualize the data

**Links**

*Data:* https://www.kaggle.com/datasets/shashwatwork/phishing-dataset-for-machine-learning/data?select=Phishing_Legitimate_full.csv


*Powerpoint:* https://1drv.ms/p/s!Aml3lvs97HXdgaBxa4hi9sMIsVEvvw?e=9cEXEI


*Colaboratory:* https://colab.research.google.com/drive/1ky1G8uxnjEQHNCgHH63nwh7pxCkGc9II?usp=sharing

***Documentation***


**1. Data Import**

**2. Preprocessing and Feature Engineering**

**3. Model Training and Evaluation**

  *Naive Bayes*

  *Bootstrap*
  
  *Bagging*
  
  *AdaBoost*
  
  *Cross-Validation*
  
**4. Visualizations**

**5. Helper Functions**
  
**Dependencies**

The project relies on the following Python libraries:

*urlopen from urllib.request*: For working with URLs.

*numpy (np alias)*: A library for numerical operations and array manipulations.

*pyplot from matplotlib*: A library for creating visualizations, used here for plotting.

*sklearn*: A machine learning library used for metrics, decision tree models, naive Bayes models, visualization, ensemble methods, and cross-validation.

***Code Structure***


**1. Data Import**

The project loads data from a file using np.loadtxt and processes it into feature and label arrays.

**2. Preprocessing and Feature Engineering**

*Feature Importance Analysis*

The code uses a decision tree to determine feature importance and selects relevant features.

*Feature Selection*

The selected features are then used to train a naive Bayes model after applying feature selection.

**3. Model Training and Evaluation**

*Naive Bayes*

The project trains a Multinomial Naive Bayes model on the selected features.

Evaluates the model using a confusion matrix, metrics, and a ROC curve.

*Bootstrap*

Utilizes bootstrap resampling to create multiple datasets.

Trains a Multinomial Naive Bayes model on each dataset.

Evaluates and visualizes the ensemble model's performance.

*Bagging*

Applies bagging by training multiple naive Bayes models on bootstrapped datasets.

Combines predictions and evaluates the ensemble model's performance.

*AdaBoost*

Uses AdaBoost to boost the performance of a decision tree.

Trains the model and evaluates its performance.

*Cross-Validation*

Evaluates naive Bayes and bootstrap models using k-fold cross-validation.

Plots the mean accuracy across different tree depths.

**4. Visualizations**

Generates visualizations, including feature importance bar plots, confusion matrices, ROC curves, and more.

**5. Helper Functions**

confusion_matrix_metrics: A helper function to display confusion matrix metrics.

**Execution**

To run the project, ensure the required dependencies are installed. You can then execute the code in a Python environment.

Note: Replace "insert file path" in the data import section with the actual file path.

***Conclusion***

This project demonstrates the application of various machine learning techniques for email phishing detection, providing insights into feature importance, model performance, and ensemble methods. Users can adapt and extend the code for similar classification tasks.
