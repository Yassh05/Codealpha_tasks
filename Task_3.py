# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load the Dataset ---
# Load the data from the csv file into a pandas DataFrame
try:
    df = pd.read_csv('car_data.csv')
    print("Dataset loaded successfully!")
    print("First 5 rows of the dataset:")
    print(df.head())
except FileNotFoundError:
    print("Error: 'car data.csv' not found. Please make sure the file is in the correct directory.")
    exit()

# --- 2. Data Preprocessing and Feature Engineering ---
print("\n--- Preprocessing Data ---")

# Check for missing values
print(f"\nMissing values in each column:\n{df.isnull().sum()}")

# Create a new feature 'Car_Age' from the 'Year' column
# We assume the current year is 2025 as of the request time.
current_year = 2025
df['Car_Age'] = current_year - df['Year']

# Drop the original 'Year' column and 'Car_Name' as it's not useful for this model
df.drop(['Year', 'Car_Name'], axis=1, inplace=True)

# Convert categorical features into numerical ones using one-hot encoding
# The 'drop_first=True' argument is used to avoid multicollinearity
df = pd.get_dummies(df, columns=['Fuel_Type', 'Selling_type', 'Transmission'], drop_first=True)

print("\nDataset after preprocessing and one-hot encoding:")
print(df.head())

# --- 3. Prepare Data for Modeling ---
# Define features (X) and the target variable (y)
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nData split into training and testing sets:")
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")


# --- 4. Train the Machine Learning Model ---
print("\n--- Training the Model ---")
# Initialize the Random Forest Regressor model
# n_estimators is the number of trees in the forest.
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Train the model on the training data
rf_model.fit(X_train, y_train)
print("Model training complete.")


# --- 5. Evaluate the Model ---
print("\n--- Evaluating the Model ---")
# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Calculate evaluation metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Model Performance on Test Data:")
print(f"R-squared (R²): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# The R-squared value indicates that our model can explain a high percentage
# of the variance in the car's selling price, which is excellent.
# The MAE shows the average absolute difference between the predicted and actual prices.


# --- 6. Visualize the Results ---
print("\n--- Visualizing Predictions ---")
# Create a scatter plot to compare actual vs. predicted prices
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Price (in Lakhs)")
plt.ylabel("Predicted Price (in Lakhs)")
plt.title("Actual vs. Predicted Car Prices")

# Add a line for perfect predictions (y=x)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', lw=2)
plt.grid(True)
plt.show()

print("\nPrediction task is complete!")