import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tpot import TPOTClassifier
from sklearn.datasets import make_classification

# Generate a random classification dataset
n_samples = 1000
n_features = 10
n_classes = 2

X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, 
                           n_informative=5, n_redundant=2, n_repeated=0, 
                           n_clusters_per_class=2, class_sep=0.8, random_state=42)

# Create feature names
feature_names = [f'feature_{i}' for i in range(n_features)]

# Create a DataFrame with the features and target
data = pd.DataFrame(X, columns=feature_names)
data['target'] = y

# Save the dataset to a CSV file
data.to_csv('synthetic_classification_data.csv', index=False)

print("Dataset generated and saved as 'synthetic_classification_data.csv'")

# Load the dataset (replace with your dataset)
data = pd.read_csv('synthetic_classification_data.csv')

# Assuming the last column is the target variable
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and run TPOT
tpot = TPOTClassifier(generations=5, population_size=20, cv=5,
                      random_state=42, verbosity=2)
tpot.fit(X_train, y_train)

# Make predictions on the test set
y_pred = tpot.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Export the best pipeline
tpot.export('tpot_best_pipeline.py')

print("\nBest pipeline:")
print(tpot.fitted_pipeline_)

# Summarize the AutoML process
print("\nAutoML Summary:")
print(f"Number of pipelines evaluated: {len(tpot.evaluated_individuals_)}")
print(f"Best pipeline score: {tpot.score(X_test, y_test):.4f}")
print(f"Best pipeline steps:")
for step in tpot.fitted_pipeline_.steps:
    print(f"- {step[0]}: {step[1]}")
