# -------------------------------------------------------------------
# TASK 1: IRIS FLOWER CLASSIFICATION
# -------------------------------------------------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Scikit-learn includes the dataset, so no download is needed
iris = load_iris()
X = iris.data  
y = iris.target  

# Create a DataFrame for better visualization and understanding
iris_df = pd.DataFrame(data=X, columns=iris.feature_names)
iris_df['species'] = y
target_names = iris.target_names
iris_df['species_name'] = iris_df['species'].apply(lambda i: target_names[i])

# Print the first 5 rows
print("--- First 5 rows of the dataset ---")
print(iris_df.head())
print("\n")

# Visualize the data with a pair plot
sns.pairplot(iris_df, hue='species_name', palette='viridis')
plt.suptitle("Pair Plot of Iris Dataset Features", y=1.02)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"--- Data Split ---")
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples\n")

# We'll use k=3 neighbors
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

print("--- Model Training ---")
print("K-Nearest Neighbors model has been trained successfully.\n")

# Make predictions on the unseen test data
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("--- Model Performance Evaluation ---")
print(f"Accuracy: {accuracy:.2f} ({accuracy*100:.0f}%)")
print("Accuracy is the percentage of correct predictions the model made.\n")

# Display the Confusion Matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("The diagonal shows correct predictions for each class (setosa, versicolor, virginica).\n")

# Visualize the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')
plt.xlabel('Predicted Species')
plt.ylabel('Actual Species')
plt.title('Confusion Matrix')
plt.show()

# Display the full Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))
print("This report shows precision, recall, and f1-score for each species.")