#load library
import pickle
import  numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
np.random.seed(42)  #constant output

#read data
ReadData=pd.read_csv(r"C:\Users\agwa8\OneDrive\Desktop\carprice\CarPrice_Assignment.csv")

#show sample data
print(ReadData.head())

#show description 
print(ReadData.describe())

#show missing data
print(ReadData.info())

#drop id car
ReadData.drop("car_ID",axis=1,inplace=True)


#convert data in coluom state and remove empty space
ReadData["CarName"]=ReadData["CarName"].str.lower()
ReadData["CarName"]=ReadData["CarName"].str.strip()
ReadData["CarName"]=ReadData["CarName"].str.split(" ",expand=True)

# get all the unique values in the 'state' column
state = ReadData['CarName'].unique()
state.sort()
print(state)


#map data
x=ReadData.drop("price",axis=1)
y=ReadData["price"]



#split data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

s = (x_train.dtypes == 'object')
object_cols = list(s[s].index)

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(x_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(x_test[object_cols]))


# One-hot encoding removed index; put it back
OH_cols_train.index = x_train.index
OH_cols_valid.index = x_test.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = x_train.drop(object_cols, axis=1)
num_X_valid = x_test.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

#choose model 
model=LinearRegression()

#train model 
model.fit(OH_X_train,y_train)

#print best train and test in model

print(model.score(OH_X_train,y_train))
print(model.score(OH_X_valid,y_test))

pickle.dump(model,open("carprise.bkl","wb"))



