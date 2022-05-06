import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
data=pd.read_csv("Chapter 02/15. TimeSeries/climate.csv")
data = data.drop(["Date Time"], axis = 1)


def pairing(data, seq_len=6):
    x = []
    y = []
    for i in range(0, data.shape[0]- (seq_len+1), seq_len+1):

        seq = np.zeros((seq_len, data.shape[1]))

        for j in range(seq_len):

            seq[j] = data.values[i+j]

        x.append(seq)
        y.append(data['Tdew (degC)'][i+seq_len])

    
    return np.array(x), np.array(y)

print(data.shape)

x,y = pairing(data)

print(x.shape)

print(y[0])
print(y[1])
print(y[2])



def new_features(data):

    
    new_feat = []
    
    for i in range(data.shape[0]):

        group = []   
    
        for j in range(data.shape[2]):

            group.append(np.std(data[i][:, j]))  
            group.append(np.mean(data[i][:, j]))
            group.append(data[i][:, j][-1])     

        new_feat.append(group)
    return np.array(new_feat)
    
        

x = new_features(x)
print(x.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


model = RandomForestRegressor()
model.fit(x_train,y_train)
pred = model.predict(x_test)
scores = cross_val_score(model, x_train,y_train, cv = 3)
print(scores)