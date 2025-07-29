# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Dataset
data = {
    "Hours": [1.5, 2.0, 3.2, 4.0, 5.5, 6.1, 7.0, 8.3],
    "Scores": [20, 25, 45, 50, 75, 80, 85, 95]
}

df = pd.DataFrame(data)

# Visualize the Data
plt.scatter(df["Hours"], df["Scores"], color='blue')
plt.title("Hours vs Scores")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.grid(True)
plt.show()

# Split Data
X = df[["Hours"]]  # Features
y = df["Scores"]   # Labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Predict for custom value
hours = 5.0
predicted_score = model.predict([[hours]])
print(f"Predicted score for {hours} hours of study: {predicted_score[0]:.2f}")

# Plot Regression Line
line = model.coef_ * X + model.intercept_
plt.scatter(X, y, color='blue')
plt.plot(X, line, color='red')  # regression line
plt.title("Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.grid(True)
plt.show()
