# Program to auto-populate matrix of data spreadsheets and identify data with separate key matrix

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

from sklearn.utils import shuffle
from keras import models
from keras import layers

# pathName = "/home/kent/LSpineDECT/"
pathName = "c:/temp"
filename = 'Low_Energy_Data.csv'

datafile = os.path.join(pathName, filename)
pre_data = pd.read_csv(datafile,index_col='Feature_Name')

#listOfFiles = list()
#listOfPAT = list()
#filenamexlsx = list()
#filenamexlsxread = 0

total_rows=len(pre_data.axes[0])
total_columns=len(pre_data.axes[1])
# len_group = slices**slices
number_of_patients = total_rows  # /len_group
valFrac = 1.0 - (number_of_patients - 1.0)/number_of_patients #validation fraction one patient
trainFrac = 1-valFrac #Training on all but one
total_Val_Acc = []
total_Val_Loss = []
test_patients = []
for i in range(int(number_of_patients)):
    data=pre_data   #  Keep original data in pre_data
    vertebrae = data.index  #  names of vertebral bodies

    testRow = data.iloc[i,:]  # test case i
    #testRow = testRow.transpose()  #  Doesn't seem to work?

    data = data.drop([data.index[i]])  #  remove the vertebral body being tested

    #test_data_df, train_data_df = np.split(data,[testRow])

    train_data_size = len(data)
    test_data_size = len(testRow)

    train_labels = copy.deepcopy(data['BMD'].astype('float32'))
    test_labels = copy.deepcopy(testRow['BMD'].astype('float32'))

    train_data = copy.deepcopy(data.iloc[:,:])
    test_data = copy.deepcopy(testRow.iloc[:])

    means = train_data.mean(axis=0)
    sigmas = train_data.std(axis=0)

    train_data = (train_data-means)/sigmas
    test_data = (test_data-means)/sigmas

    model = models.Sequential()
    model.add(layers.Dense(8, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(8, activation='relu', input_shape=(train_data.shape[1],)))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    history = model.fit(train_data,train_labels, epochs=20, batch_size=2000, validation_data=(test_data, test_labels))
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)


    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']
    acc_values = history_dict['acc']
    last_val_acc_value = val_acc_values[-1]
    last_val_loss_value = val_loss_values[-1]
    print(test_data_df.index[0])
    print(last_val_acc_value)
    print(last_val_loss_value)
    test_patients.append(test_data_df.index[0])
    total_Val_Acc.append(last_val_acc_value)
    total_Val_Loss.append(last_val_loss_value)

    last99Patient_cycle_data = pre_data[total_rows - (total_rows-len_group):]
    first1percent_cycle_data = pre_data[:total_rows - (total_rows-len_group)]
    pre_data = pd.concat([last99Patient_cycle_data, first1percent_cycle_data])
print(np.mean(total_Val_Acc))
print(np.mean(total_Val_Loss))
print(total_Val_Acc)
print(total_Val_Loss)
print(test_patients)
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.clf()
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
