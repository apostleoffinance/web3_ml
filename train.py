import numpy as np
import pandas as pd 
import plotly.graph_objects as go
import random
import os
#import pickle
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score

model_file = 'logistic_regression_model.bin'

""" Get your data as a pandas data frame"""

df = pd.read_csv('traderclassifier.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')


#Set the display option to show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)  # Adjusts the display width for better visibility


# Convert 'day' column to datetime format and remove time component 
df['day'] = pd.to_datetime(df['day'], errors='coerce').dt.date

# Remove outliers from 'portfolio_return' column based on IQR method
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Apply the function to remove outliers from the 'portfolio_return' column
df_clean = remove_outliers_iqr(df, 'portfolio_return')


#drop rows in trader_class column with NaN values
df_cleann = df_clean.dropna(subset=['trader_class'])
#Fill NaN with 0
df_cleaned = df_cleann.fillna(0)

#Datatypes

#convert day to datetime
df_cleaned['day'] = pd.to_datetime(df_cleaned['day'])

# Convert 'trader_class' to a categorical column
df_cleaned["trader_class"] = df_cleaned["trader_class"].astype('category')
# Assign unique integers to categories
df_cleaned['trader_class_numeric'] = df_cleaned['trader_class'].astype('category').cat.codes

print(df_cleaned.head())

# Correlation Coefficient Analysis

df_cor = df_cleaned.corr(numeric_only=True)
print(df_cor['trader_class_numeric'].sort_values(ascending=False))

numeric = df_cleaned.select_dtypes(include=['number']).drop(columns=['trader_class_numeric'], errors='ignore')
# Reset the index to ensure it is not included as a column
features = numeric.reset_index(drop=True)
features = [col for col in features if col in df_cleaned.columns]

target = 'trader_class_numeric'

#Data Processing
#Separating the features from the target variable.

X = df_cleaned[features]
y = df_cleaned[target]

#Normalize the Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Train the logistic Regression Model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
model.fit(X_train, y_train)

#predict on the test set
y_pred = model.predict(X_test)

# Save the Model
# Save the trained model to a .bin file

# with open(model_file, 'wb') as f_out:
#     pickle.dump((scaler, model), f_out)

# print(f"Model saved to {model_file}")

# Save the scaler and model
model_file = 'logistic_regression_model.bin' 
joblib.dump((scaler, model), model_file)

print(f"Model saved to {model_file}")




