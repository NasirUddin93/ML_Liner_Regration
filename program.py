import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("price.csv")
print(df)

reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)
print(reg.predict([[3300]]))
print(reg.coef_)
print(reg.intercept_)

plt.xlabel('area(sqr ft',fontsize=20)
plt.ylabel('price(US$)',fontsize=20)
plt.scatter(df.area, df.price, color='red', marker=('+'))
plt.plot(df.area,reg.predict(df[['area']]),color='blue')



d = pd.read_csv("areas.csv")
print(d.head(3))
p= reg.predict(d)
d['prices'] = p
print(d)
d.to_csv("prediction.csv")


