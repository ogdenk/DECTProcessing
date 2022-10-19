import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
# import tensorflow as tf
from keras import models
from keras import layers

#pathName = "/home/kent/LSpineDECT/"
pathName = "c:/temp"
filename = 'Low_Energy_Data.csv'

datafile = os.path.join(pathName, filename)
pre_data = pd.read_csv(datafile,index_col='Feature_Name')

total_rows=len(pre_data.axes[0])
total_columns=len(pre_data.axes[1])
len_group = 1  # slices**slices
number_of_patients = total_rows/len_group
valFrac = 1.0 - (number_of_patients - 1.0)/number_of_patients # validation fraction one patient
trainFrac = 1-valFrac # Training on all but one
total_Val_Acc = []
total_Val_Loss = []
test_patients = []
trialsPerPatient = 5

for i in range(int(number_of_patients)):
    data_df = pre_data   # Keep original data in pre_data
    vertebrae = data_df.index  # names of vertebral bodies

        #         Same as df.iloc[[i],:]
    testRow_df = data_df.iloc[[i]]  # test case i  Use double brackets to get a row back
    shape = testRow_df.shape

    data_df = data_df.drop([data_df.index[i]])  # remove the vertebral body being tested

        # test_data_df, train_data_df = np.split(data,[testRow])
    train_data_df = copy.deepcopy(data_df.iloc[:, :])  # Same as data_df
    test_data_df = copy.deepcopy(testRow_df.iloc[:])  # Same as testRow_df

    train_data = train_data_df.to_numpy()[:, 1:]  # Convert to numpy arrays, drop BMD values
    test_data = test_data_df.to_numpy()[:, 1:]

    train_data_size = len(train_data)  # How big are these arrays?
    test_data_size = len(test_data)

    train_labels = copy.deepcopy(data_df['BMD'].to_numpy())
    test_labels = copy.deepcopy(test_data_df['BMD'].to_numpy())

    means = train_data.mean(axis=0)
    sigmas = train_data.std(axis=0)

    train_data = (train_data - means) / sigmas
    test_data = (test_data - means) / sigmas

    train_data = np.asarray(train_data)
    train_labels = np.asarray(train_labels)
    test_data = np.asarray(test_data)
    test_labels = np.asarray(test_labels)

    for j in range(trialsPerPatient):

        model = models.Sequential()
        model.add(layers.Dense(8, activation='relu', input_shape=(train_data.shape[1],)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(8, activation='relu'))
        # model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='linear'))
        model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['mean_squared_error'])

        history = model.fit(train_data, train_labels, epochs=5, batch_size=15, validation_data=(test_data, test_labels))
        history_dict = history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epochs = range(1, len(loss_values) + 1)


        MSE_values = history_dict['mean_squared_error']
        val_MSE_values = history_dict['val_mean_squared_error']
        last_val_MSE_value = val_MSE_values[-1]
        last_val_loss_value = val_loss_values[-1]
        print(test_data_df.index[0])
        print(last_val_MSE_value)
        print(last_val_loss_value)
        test_patients.append(test_data_df.index[0])
        total_Val_Acc.append(last_val_MSE_value)
        total_Val_Loss.append(last_val_loss_value)

        #last99Patient_cycle_data = pre_data[total_rows - (total_rows-len_group):]
        #first1percent_cycle_data = pre_data[:total_rows - (total_rows-len_group)]
        #pre_data = pd.concat([last99Patient_cycle_data, first1percent_cycle_data])

print(np.mean(total_Val_Acc))
print(np.mean(total_Val_Loss))
print(total_Val_Acc)
print(total_Val_Loss)
print(test_patients)
plt.plot(epochs, MSE_values, 'bo', label='Training MSE')
plt.plot(epochs, val_MSE_values, 'b', label='Validation MSE')
plt.title('Training and validation MSE')
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
