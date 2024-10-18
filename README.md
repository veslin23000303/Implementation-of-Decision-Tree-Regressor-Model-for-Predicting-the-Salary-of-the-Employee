# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start the program
Step 2: Import the required libraries.
Step 3: Upload the csv file and read the dataset.
Step 4: Check for any null values using the isnull() function.
Step 5: From sklearn.tree import DecisionTreeRegressor.
Step 6: Import metrics and calculate the Mean squared error.
Step 7: Apply metrics to the dataset, and predict the output.
Step 8: End the program
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: VESLIN ANISH A
RegisterNumber:212223240175
import pandas as pd
data = pd.read_csv("Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()

x = data[["Position","Level"]]
y = data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =
train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse

r2 = metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
*/
```

## Output:

![Screenshot 2024-10-18 113918](https://github.com/user-attachments/assets/bd33d96c-5e97-4f27-b9fa-8cec3d1e7c71)

![Screenshot 2024-10-18 113923](https://github.com/user-attachments/assets/ca9e0c28-773c-4c66-ba9e-42bf08024b42)

![Screenshot 2024-10-18 113928](https://github.com/user-attachments/assets/fd5b7a40-1173-4a04-85f1-e0c527fd1c1b


![Screenshot 2024-10-18 113931](https://github.com/user-attachments/assets/6f757fde-90e0-47c6-a724-8630e969609e)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
