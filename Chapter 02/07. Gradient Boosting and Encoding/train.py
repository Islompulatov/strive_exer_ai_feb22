import data_handler as dh
import numpy
x_train, x_test, y_train, y_test = dh.get_data("Chapter 02/07. Gradient Boosting and Encoding/insurance.csv")

from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators=100, criterion='mae')




clf.fit(x_train, y_train)
# make predictions
predictions = clf.predict(x_test)
scores = (predictions == y_test).sum()/len(y_test)
print(scores)
print(dh.hello)