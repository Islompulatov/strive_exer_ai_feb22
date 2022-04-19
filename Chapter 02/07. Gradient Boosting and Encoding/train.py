import data_handler as dh
import numpy
x_train, x_test, y_train, y_test = dh.get_data("Chapter 02/07. Gradient Boosting and Encoding/insurance.csv")

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
clf = RandomForestRegressor(n_estimators=100, criterion='mae')


x_train_scaled = scaler.fit_transform(x_train,y_train)
x_test_scaled = scaler.transform(x_test)

clf.fit(x_train_scaled, y_train)
# make predictions
predictions = clf.predict(x_test_scaled)
scores = (predictions == y_test).sum()/len(y_test)
print(scores)
print(dh.hello)