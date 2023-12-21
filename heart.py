# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset 
dataset = pd.read_csv('heart.csv')

#One-hot encoding converts categorical data into a binary format
onehot_encoded = pd.get_dummies(dataset['Sex'], prefix='Sex')
dataset = pd.concat([dataset, onehot_encoded], axis=1)
onehot_encoded = pd.get_dummies(dataset['ChestPainType'], prefix='ChestPainType')
dataset = pd.concat([dataset, onehot_encoded], axis=1)
onehot_encoded = pd.get_dummies(dataset['RestingECG'], prefix='RestingECG')
dataset = pd.concat([dataset, onehot_encoded], axis=1)
onehot_encoded = pd.get_dummies(dataset['ExerciseAngina'], prefix='ExerciseAngina')
dataset = pd.concat([dataset, onehot_encoded], axis=1)
onehot_encoded = pd.get_dummies(dataset['ST_Slope'], prefix='ST_Slope')
dataset = pd.concat([dataset, onehot_encoded], axis=1)

# Separate Input and Output Variables
# Remove excessive columns left after One-hot encoding in input
X = dataset.drop(['HeartDisease', 'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], axis=1)
y = dataset['HeartDisease']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train)

# Classification using k-NN
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)

print("k-NN Accuracy:", knn_accuracy)
print("k-NN Classification Report:")
print(classification_report(y_test, knn_predictions))

# Classification using Decision Trees
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
dt_predictions = dt_classifier.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

print("Decision Trees Accuracy:", dt_accuracy)
print("Decision Trees Classification Report:")
print(classification_report(y_test, dt_predictions))