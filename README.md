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

**Documentation**
