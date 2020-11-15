
import pandas as pd 
import numpy as np
import pickle
from matplotlib import pyplot as plt
import io
from sklearn.linear_model import LinearRegression

df = pd.read_csv('C:/Fall20/Data Mining/true_car_listings.csv')

linearmodel = LinearRegression()

X=df[['Year','Mileage']]
y=df['Price']
#y.head()
linearmodel.fit(X, y)

pickle.dump(linearmodel, open('linear.pkl','wb'))

model = pickle.load(open('linear.pkl','rb'))
#print(model.predict([[4, 300, 500]]))