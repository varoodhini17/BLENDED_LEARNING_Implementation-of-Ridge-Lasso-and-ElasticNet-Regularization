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
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,r2_score
data=pd.read_csv("encoded_car_data (1).csv")
data.head()
data=pd.get_dummies(data,drop_first=True)
X=data.drop('price',axis=1)
y=data['price']
scaler=StandardScaler()
X=scaler.fit_transform(X)
y=scaler.fit_transform(y.values.reshape(-1,1))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
models={
    "Ridge":Ridge(alpha=1.0),
    "Lasso":Lasso(alpha=1.0),
    "ElectricNet":ElasticNet(alpha=1.0,l1_ratio=0.5)
}
results={}
for name,model in models.items():
    pipeline=Pipeline([
        ('poly',PolynomialFeatures(degree=2)),

     ('regressor',model)
    ])
pipeline.fit(X_train,y_train)
predictions=pipeline.predict(X_test)
mse=mean_squared_error(y_test,predictions)
r2=r2_score(y_test,predictions)
results[name]={'MSE':mse,'R2 Score':r2}
print('Name: Varoodhini.M')
print('Reg. No:212225220118')
for model_name,metrics in results.items():
    print(f"{model_name}-Mean Squared Error: {metrics['MSE']:.2f},R2 Score: {metrics['R2 Score']:.2f}")
results_df=pd.DataFrame(results).T
results_df.reset_index(inplace=True)
results_df.rename(columns={'index':'Model'},inplace=True)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.barplot(x='Model',y='MSE',data=results_df,palette='viridis')
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')
plt.xticks(rotation=45)
plt.subplot(1,2,2)
sns.barplot(x='Model',y='R2 Score',data=results_df,palette='viridis')
plt.title('R2 Score')
plt.ylabel('R2 Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

```

## Output:
<img width="529" height="75" alt="image" src="https://github.com/user-attachments/assets/db4f4dbb-4a10-43c9-8014-8ae8a4976935" />

<img width="411" height="678" alt="image" src="https://github.com/user-attachments/assets/19f7a6ae-bf5b-4ee8-8e62-4b02e54edadf" />

<img width="394" height="690" alt="image" src="https://github.com/user-attachments/assets/f9849a29-690d-4b51-b5a2-cd7dc486311a" />


## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
