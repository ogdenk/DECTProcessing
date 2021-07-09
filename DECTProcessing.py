# Program to auto-populate matrix of data spreadsheets and identify data with separate key matrix

import os
import numpy as np
import pandas as pd
import h5py
h5py.run_tests()

from sklearn.utils import shuffle
from keras import models
from keras import layers
import matplotlib.pyplot as plt


pathName = "X:\Research\DECT BMD Study"

slices = 4  #slices set as 1 for seizure study.  use 2 for no data augmentation tumors. 4 for data augmentation tumors

rootDir = 'c:/temp'
#filename = 'rawavg_nogad1.5.csv'
#filename = 'dataSetGBM2noADC.csv'
#filename = 'dataSetEmb2.4.19noDataAugFeatureAvg.csv' # 1 x 4
#filename = 'dataSetEmb2.4.19noDataAug.csv' # 1 x 16,
#filename = 'dataSetEmb2.4.19noDataAugFeatureAvg.csv' # 1 x 16
filename='dataSetEmb1.30.19.csv'
#filename='dataSetEmb2.12.19noDataAug1.csv'
#filename = 'dataSetEmb1.30.19.csv' # Fully augmented

datafile = os.path.join(rootDir, filename)
pre_data = pd.read_csv(datafile,index_col='Patient_Number')

#########################################################################################################

#f = h5py.File('d:/dect/myfile.hdf5')

listOfFiles = list()
listOfPAT = list()
filenamexlsx = list()
filenamexlsxread = 0

# use os.walk() to walk through directory and grab files that we're interested in
for root, dirs, files in os.walk(pathName, topdown=True):
    filenamexlsx = [file for file in files if file.endswith('.xlsx')]

    # First read in the xlsx data file and save to hdf5 file
    if filenamexlsx != []:
        if filenamexlsxread == 0:
            filenamexlsxread = 1
            xlFile = root + "/" + filenamexlsx[0]
            xldata = pd.read_excel(xlFile, sheet_name='Sheet1')

         # write data from xlsx to the output file
    #directories = [d for d in dirs if d.startswith('DECT')]  # only look in folders that start with dect
    filename = [file for file in files if file.endswith('.dcm')]  # only grab .dcm files (all we need)

    patient = root
    patient = patient[-6:]

    length = len(filename)
    count = 0
    while count < length:
        outputfilename = patient + "-" + str(filename[count])[-8:]
        count = count+1

    listOfFiles += [os.path.join(root, filename) for filename in files]

# remove pathName and filename from root list, this gives us a list of patient names with one per file

print("Done!")
###########################################################################################################################




total_rows=len(pre_data.axes[0])
total_columns=len(pre_data.axes[1])
len_group = slices**slices
number_of_patients = total_rows/len_group
valFrac = 1-(number_of_patients-1)/number_of_patients #Training fraction all but one patient
trainFrac = 1-valFrac #Validation one patient
total_Val_Acc = []
total_Val_Loss = []
test_patients = []
for i in range(int(number_of_patients)):
    #last256_cycle_data = pre_data.as_matrix()[(total_rows-len_group):,:]

    #index_list = np.array(pre_data.index)
    #np.reshape(index_list, (-1, len_group))
    data=pre_data

    #data2=pd.DataFrame(data,index=index_list)
    #data3=data.loc['Tumor Type', 'Patient Number']
    #data = data.loc[index_list, :]
    #data = shuffle(pre_data)
    testRows = slices**slices
    #testRows = int(valFrac*data.shape[0]+1)  #add 1 due to computer rounding error in some data sets.  make sure its 256 if this is happening

    test_data_df, train_data_df = np.split(data,[testRows])

    train_data_size = len(train_data_df)
    test_data_size = len(test_data_df)

    train_labels = train_data_df['Tumor_Type'].as_matrix().astype('float32')
    test_labels = test_data_df['Tumor_Type'].as_matrix().astype('float32')
    #train_labels = train_data_df['SzBd400'].as_matrix().astype('float32')
    #test_labels = test_data_df['SzBd400'].as_matrix().astype('float32')

    train_data = train_data_df.as_matrix()[:,3:]
    test_data = test_data_df.as_matrix()[:,3:]


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
