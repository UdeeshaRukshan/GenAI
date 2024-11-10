import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Create a simple dataset
data = {
    'age': [25, 32, 47, 51, 23, 35, 52, 46, 44, 36],
    'income': [50000, 60000, 80000, 82000, 48000, 54000, 92000, 78000, 75000, 62000],
    'purchased': [0, 1, 1, 1, 0, 0, 1, 1, 1, 0]  # 1 means purchased, 0 means not purchased
}

# Convert to DataFrame
df = pd.DataFrame(data)
print(df.head())

X = df[['age', 'income']]  # Features
y = df['purchased']        # Target (what we are predicting)

# Split the dataset into training and testing data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Measure the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

print("Model Predictions:", model.predict(X_test))
print("Actual Test Values:", y_test)

plt.scatter(df['age'], df['income'], c=df['purchased'], cmap='coolwarm')
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Customer Purchases (0: Not Purchased, 1: Purchased)')
plt.show()
 