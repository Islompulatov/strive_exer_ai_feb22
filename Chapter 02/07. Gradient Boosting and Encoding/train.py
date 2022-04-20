from time import process_time_ns
from wsgiref.util import request_uri

import joblib
import data_handler as dh
from sklearn.metrics import accuracy_score
x_train, x_test, y_train, y_test = dh.get_data("Chapter 02/07. Gradient Boosting and Encoding/insurance.csv")

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor,GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


# clf = RandomForestRegressor(n_estimators=100, criterion='mae')
# adbr = AdaBoostRegressor(n_estimators=50, learning_rate=1, loss = 'linear')
# gdbr = GradientBoostingRegressor(n_estimators=100, verbose=0)


class Normal:
    def __init__(self, x_train, x_test, y_train, y_test):    
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    
    # def standart_scaler(self):
    #     self.scaler = StandardScaler()
    #     self.x_train_scaled = self.scaler.fit_transform(x_train,y_train)
    #     self.x_test_scaled = self.scaler.transform(x_test)
    #     return self.x_train_scaled, self.x_test_scaled
    # models = [RandomForestRegressor(), AdaBoostRegressor(), GradientBoostingRegressor()]
    def prediction():
        rmfr  = RandomForestRegressor(n_estimators=100, criterion='mae')
        rmfr.fit(x_train,y_train)
        predictions = rmfr.predict(x_test,y_test)
        scores = rmfr.score(predictions,y_test)
    
        return predictions, scores

        
    

class Models(Normal):
    def __init__(self, x_train, x_test, y_train, y_test):
        super().__init__(x_train, x_test, y_train, y_test)


    def ada_boost(self):
        model  = AdaBoostRegressor(n_estimators=50, learning_rate=1, loss = 'linear')
        model.fit(x_train,y_train)
        predictions = model.predict(x_test,y_test)

        scores = self.model.score(predictions,y_test)

        return predictions, scores
      

    def gradient_boost(self):
        self.model  = GradientBoostingRegressor(n_estimators=100, verbose=0)
        self.model.fit(x_train,y_train)
        predictions = self.model.predict(x_test,y_test)

        scores = self.model.score(predictions,y_test)

        return predictions, scores


    # def fit_models():
    #     models = [RandomForestRegressor(), AdaBoostRegressor(), GradientBoostingRegressor()]

    #     fitted_model = []
    #     for i in range(len(models)):
    #         fitted_model.append(models[i].fit(x_train,y_train))
    #     return fitted_model
        

    # def predicts(self):
    #     models = [RandomForestRegressor(), AdaBoostRegressor(), GradientBoostingRegressor()]
    #     predictions = []
    #     for j in range(len(models()):
    #         predictions.append(self.models[j].predict(x_test,y_test))
    #     return predictions    
            
            # predictions.append(models[i].predict(x_test,y_test))
            

"""
1. Make list of models
2.Loop through the list
3.Fit every model
4.append a fitted model in a new list
5.return new list
6.make a new function for predict the model
7.Loop through the new list form the step 4
8. Call predict for each model
9.append the predictions to new a list 'predictions'
10.return predictions
"""


    # def ada_model(self):
        
    #     model = AdaBoostRegressor(n_estimators=50, learning_rate=1, loss = 'linear')
    #     model.fit(self.x_train_scaled,y_train)
    #     acc = model.score(self.x_test_scaled,y_test)
    #     print(f'Adaboost score: ' +{acc})
    #     return model
  


    # def gradient_model(self):
    #     model = GradientBoostingRegressor(n_estimators=100, verbose=0)
    #     model.fit(self.x_train_scaled, y_train)
    #     acc1 = model.score(self.x_test_scaled,y_test)
    #     print(f'Gradientboost score: ' + {acc1})
    #     return model
        
    


print(dh.hello)