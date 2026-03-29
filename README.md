# BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and convert categorical data into numerical form using one-hot encoding.

2.Separate input and output variables, then scale the data using StandardScaler.

3.Split the dataset into training and testing sets (80% training, 20% testing).

4.Apply Polynomial Regression with Ridge, Lasso, and ElasticNet models using a pipeline and train them.

5.Evaluate the models using Mean Squared Error (MSE) and R² score, then compare the results using graphs. 

## Program:
```
/*
Program to implement Ridge, Lasso, and ElasticNet regularization using pipelines.
Developed by: Varoodhini.M
RegisterNumber: 212225220118  
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
data=pd.read_csv("encoded_car_data (1).csv")
data.head()
data = pd.get_dummies(data, drop_first=True)
x=data.drop('price',axis=1)
y=data['price']
scaler = StandardScaler()
x = scaler.fit_transform(x)
y = scaler.fit_transform(y.values.reshape(-1, 1))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
models = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
    "ElasticNet": ElasticNet (alpha=1.0, l1_ratio=0.5)
}
result= {}
for name,model in models.items():
    pipeline = Pipeline([('poly',PolynomialFeatures(degree=2)),
    ('regressor',model)
    ])
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    result[name] = {'MSE': mse, 'MAE': mae, 'R² Score': r2}
print('Name: VAROODHINI M')
print('Reg. No: 212225220118')
for model_name, metrics in result.items():
    print (f"{model_name} - Mean Squared Error: {metrics['MSE']:.2f}, Mean Absolute Error: {metrics['MAE']:.2f}, R² Score: {metrics['R² Score']:.2f}")
results_df = pd.DataFrame(result).T
results_df.reset_index(inplace=True)
results_df.rename(columns={'index': 'Model'}, inplace=True)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='MSE', data=results_df, palette='viridis')
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')
plt.xticks(rotation=45)
plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='R² Score', data=results_df, palette='viridis')
plt.title('R2 Score')
plt.ylabel('R2 Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

```

## Output:
<img width="887" height="146" alt="image" src="https://github.com/user-attachments/assets/7df9bdfc-37f8-4f4e-ab3b-6632a8371164" />

<img width="1259" height="535" alt="image" src="https://github.com/user-attachments/assets/ea7b9b89-4626-4aeb-b87c-fb84c5ae22e8" />






## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
