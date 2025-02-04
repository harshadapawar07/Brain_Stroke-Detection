import pandas as pd
import numpy as np

print("version",np.version)


np.random.seed(42)
n_samples=1000
data = {
    'Age': np.random.randint(0, 80, n_samples),
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'BP': np.random.uniform(90, 120, n_samples),                                       #blood pressure 
    'Smoking Status': np.random.choice(['Never', 'Former', 'Current'], n_samples),
    'Physical Activity': np.random.choice(['Sedentary', 'Moderate', 'Active'], n_samples),
    'ICP':np.random.randint(5,15,n_samples),                                               #intracranial pressure
    'CBF':np.random.randint(50,70,n_samples),                                              #cerebral blood flow 
    'BT':np.random.randint(35,37,n_samples),                                               # body temperature 
    'CPP':np.random.randint(60,100,n_samples),                                             #cerebral perfusion presuure 
    'WBC':np.random.randint(4500,11000,n_samples) ,                                         #white blood cell
    'Stroke': np.random.choice([0, 1], n_samples)
    
}

df = pd.DataFrame(data)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

encoder = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(), ['Gender', 'Smoking Status', 'Physical Activity'])
], remainder='passthrough')

x = encoder.fit_transform(df[['Age', 'BP', 'ICP', 'CBF', 'BT', 'CPP', 'WBC', 'Gender', 'Smoking Status', 'Physical Activity']])


y=df['Stroke']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Import necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the logistic regression model
LR = LogisticRegression()

# Train the model
LR.fit(x_train, y_train)

# Make predictions
y_pred = LR.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import pickle
from sklearn.preprocessing import StandardScaler

# Assuming this is the scaler used during training
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Save the scaler to a file
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Save the model as well
with open('brain_stroke.pkl', 'wb') as model_file:
    pickle.dump(LR, model_file)
