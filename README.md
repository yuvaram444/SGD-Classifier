# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize Parameters: Set initial weights (theta) to zero.
2. Compute Predictions: Calculate predictions using the sigmoid function on the weighted inputs.
3. Calculate Cost: Compute the cost using the cross-entropy loss function.
4. Update Weights: Adjust weights by subtracting the gradient of the cost with respect to each weight.
5. Repeat: Repeat steps 2–4 for a set number of iterations or until convergence is achieved.

## Program:
```
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Yuvaram S
RegisterNumber:  212224230315
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df.head())

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train, y_train)

# Predictions
y_pred = sgd_clf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    fmt='d',
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

```

## Output:
<img width="897" height="793" alt="image" src="https://github.com/user-attachments/assets/e2578bd6-08eb-4db5-b05e-7c302b04d99e" />





## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
