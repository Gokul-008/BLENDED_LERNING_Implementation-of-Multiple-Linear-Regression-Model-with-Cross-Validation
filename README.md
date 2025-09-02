# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Load and Prepare Data**  
   - Load the dataset using Pandas.  
   - Drop unnecessary columns like `car_ID` and `CarName`.  
   - Convert categorical variables into numerical format using one-hot encoding.

2. **Split the Data**  
   - Separate the dataset into features (`X`) and target variable (`y`).  
   - Split the dataset into training and testing sets using `train_test_split`.

3. **Build and Train the Model**  
   - Create a `LinearRegression` model instance.  
   - Fit the model on the training data.

4. **Evaluate the Model**  
   - Perform 5-fold cross-validation using `cross_val_score`.  
   - Evaluate the model on the test set using Mean Squared Error (MSE) and R² score.  
   - Visualize the actual vs predicted car prices using a scatter plot.

## Program:
```

Program to implement the multiple linear regression model for predicting car prices with cross-validation.
Developed by: GOKUL M
RegisterNumber:  212222230037

```
```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv('/content/CarPrice_Assignment (1).csv')

data = data.drop(['car_ID','CarName'],axis=1)
data = pd.get_dummies(data,drop_first=True)

x = data.drop('price',axis=1)
y = data['price']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(x_train,y_train)

print('GOKUL M')
print('212222230037')
print("\n===Cross Validation ===")
cv_scores = cross_val_score(model, x, y, cv=5)
print("Fold R² Scores:",[f"{score:.4f}" for score in cv_scores])
print(f"Average R² Score: {cv_scores.mean():.4f}")

y_pred = model.predict(x_test)
print("\n===Test Set Perfomance ===")
print(f"MSE:{mean_squared_error(y_test,y_pred):.2f}")
print(f"R²: {r2_score(y_test,y_pred):.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()],'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actua1 vs Predicted Prices")
plt .grid(True)
plt. show( )

```


## Output:
<img width="928" height="246" alt="Screenshot 2025-09-03 003236" src="https://github.com/user-attachments/assets/f65b6ee3-6ebe-4ffa-8e66-da1f99aeb794" />

<img width="1170" height="637" alt="Screenshot 2025-09-03 003300" src="https://github.com/user-attachments/assets/37f6a3c0-4942-4b35-86a1-527862fb6e77" />


## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
