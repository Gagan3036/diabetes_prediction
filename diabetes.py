import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
diabetes = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Diabetes.csv')

# Prepare the data
y = diabetes['diabetes']
X = diabetes[['pregnancies', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi', 'dpf', 'age']]  

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2529) 

# Train the model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, r'C:\Users\gagan\Desktop\diabetes_prediction\diabetes_prediction_model.pkl')

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)
print(f'Model accuracy: {accuracy}')
