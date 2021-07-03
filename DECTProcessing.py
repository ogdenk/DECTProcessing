# Program to auto-populate matrix of data spreadsheets and identify data with separate key matrix

#test

import os

import numpy as np
import pandas as pd
import h5py
h5py.run_tests()

pathName = "X:\Research\DECT BMD Study"


f = h5py.File('d:/dect/myfile.hdf5')

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
