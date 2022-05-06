import pandas as pd 
import numpy as np
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



def getfeatures(data):

    
    new_data = []
    
    for i in range(data.shape[0]):

        group = []   
    
        for j in range(data.shape[2]):

            group.append(np.mean(data[i][:, j]))  
            group.append(np.std(data[i][:, j]))
            group.append(data[i][:, j][-1])     

        new_data.append(group)

    return np.array(new_data)


new_x = getfeatures(x)