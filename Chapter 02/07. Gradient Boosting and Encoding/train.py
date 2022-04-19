import data_handler as dh

x_train, x_test, y_train, y_test = dh.get_data("./insurance.csv")

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, criterion='gini')




clf.fit(x_train, y_train)
# make predictions
predictions = clf.predict(x_test)
scores = (predictions == y_test).sum()/len(y_test)
print(dh.hello)