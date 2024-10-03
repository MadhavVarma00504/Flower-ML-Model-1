# Flower-ML-Model-1

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score




# Load the iris dataset
data, target = load_iris(return_X_y=True)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=45)

# Initialize the model (Support Vector Machine)
model = SVC()

# Train the model
model.fit(x_train, y_train)

# Test the model and calculate accuracy score
predictions = model.predict(x_test)
accuracy = accuracy_score(y_test, predictions)

print(accuracy)
