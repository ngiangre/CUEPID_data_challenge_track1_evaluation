import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from keras.layers import Dropout

import warnings
warnings.simplefilter('ignore')

recoveredData = pd.read_csv("time_series_covid19_recovered_global.csv")

numOfDays = (recoveredData.shape[1]-4)
numOfCountries = (recoveredData.shape[0])
may1st=8 # Days until May1st
win = 60 #window for the previous n days

recovered = np.empty((0,numOfDays))
for index, row in recoveredData.head(n=300).iterrows():
    singleRow = np.empty(0)
    for i in recoveredData.columns:
        if (i=='Province/State')|(i=='Country/Region')|(i=='Lat')|(i=='Long'):
            continue
        singleRow = np.concatenate((singleRow,np.array(row[i]).reshape(1,)),0)

    recovered =  np.concatenate((recovered,singleRow.reshape(1,numOfDays)),0)


may1stRecovered = np.zeros(numOfCountries)

for index in range(numOfCountries):
    print("************index: ",index)

    recoveredTarget = np.zeros(recovered[index].shape[0]-win)

    tryData = np.empty((0,win,1))
    for i in range(recovered[index].shape[0]-(win)):
        temp = np.empty((0,win))
        temp = np.concatenate((temp,recovered[index][i:i+win].reshape(1,win)),0)
        temp=temp.T

        tryData = np.concatenate((tryData,temp.reshape(1,win,1)),0)
        recoveredTarget[i] = recovered[index][i+win]

    finalData = tryData[:]
    finalRecoveredTarget = recoveredTarget[:]


    #####################################
    clf = MLPRegressor(hidden_layer_sizes = (100000),activation = 'relu',solver = 'adam',alpha = 0.001,random_state = 7)

    clf.fit(finalData.reshape(recovered[index].shape[0]-win,win),finalRecoveredTarget.reshape(recovered[index].shape[0]-win,1))

    generatedDataMLP = finalData.reshape(recovered[index].shape[0]-win,win)
    for i in range(may1st):
        singleRecoveredResultMLP = clf.predict(generatedDataMLP[-1].reshape(1,win))
        temp1 = np.empty((0,1))
        temp1 = np.concatenate((temp1,singleRecoveredResultMLP.reshape(1,1)),0)
        temp1 = temp1.T
        temp2 = np.concatenate((generatedDataMLP[-1].reshape(win,1)[-(win-1):],temp1),0)
        generatedDataMLP = np.concatenate((generatedDataMLP,temp2.reshape(1,win)),0)


    #########################################

    may1stRecovered[index] = generatedDataMLP.T[-1][-1]

    #print(may1stRecovered[index])

for i in may1stRecovered:
    print(int(i))
