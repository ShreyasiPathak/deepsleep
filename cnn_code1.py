# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:47:16 2018

@author: pathaks

Convolutional Neural Network for sleep stage detection
"""

import os
import scipy.io as sio
import numpy as np
from random import shuffle
from keras.models import Sequential
from keras.utils import np_utils, Sequence
from keras.layers import Conv1D, Dense, Dropout, MaxPooling1D
from keras.layers import Reshape, Activation, Flatten

'''class MY_Generator(Sequence):
    def __init__(self, mat_files, batch_size, n_classes):
        self.mat_files= mat_files
        self.batch_size = batch_size
        self.n_classes=n_classes

    def __len__(self):
        return np.ceil(len(self.mat_files) / float(self.batch_size))

    def __getitem__(self, idx):
        batch_filenames = self.mat_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        #batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        X,y=self._data_generation(batch_filenames)'''
        
    
def generator(batch_size, mat_files, time_period_sample, n_classes, n_channels,batches_per_epoch):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    # Initialization
    batch_start=0
    batch_end=batch_size
    k=0
    batch_no=0
    batches=0
    global_batch=0
    
    while 1:
        if global_batch==batches_per_epoch:
            k=0
            global_batch=0
        
        # Store sample
        if batch_no>=batches:
            x_epochs = np.empty((batch_size, time_period_sample, n_channels))
            y_epochs = np.empty((batch_size), dtype=int)
            data = sio.loadmat(path_to_mat_folder+"/"+mat_files[k],struct_as_record=False)
            print(mat_files[k])
            eeg1=data['signals'][0,0].eeg1[0]
            eeg2=data['signals'][0,0].eeg2[0]
            y=data['signals'][0,0].annotation[0]
            #print(len(eeg1)/(time_period_sample))
            #print(time_period_sample)
            batches=np.ceil(len(eeg1)/(time_period_sample*batch_size))
            #print("batches:"+str(batches))
            batch_start=0
            batch_end=batch_size
            batch_no=0
            k=k+1
            for j in range(len(y)):
                if y[j]==5:
                    y[j]=4
        
        else:
            if batch_no==batches-1:
                batch_start=batch_end
                batch_end=int(len(eeg1)/(time_period_sample))
                x_epochs = np.empty((batch_end-batch_start, time_period_sample, n_channels))
                y_epochs = np.empty((batch_end-batch_start), dtype=int)
            else:
                batch_start=batch_end
                batch_end=batch_start+batch_size
                x_epochs = np.empty((batch_size, time_period_sample, n_channels))
                y_epochs = np.empty((batch_size), dtype=int)
        
        # Generate data of batch size
        m=0
        for i in range(batch_start,batch_end):
            if y[i*time_period_sample] in [0,1,2,3,4]:
                eeg1_epoch=eeg1[i*time_period_sample:(i+1)*time_period_sample]
                eeg2_epoch=eeg2[i*time_period_sample:(i+1)*time_period_sample]
                x_epochs[m]=np.asarray([eeg1_epoch,eeg2_epoch]).reshape(time_period_sample,n_channels)
                y_epochs[m]=np.unique(y[i*time_period_sample:(i+1)*time_period_sample])
                m=m+1
            else:
                x_epochs=np.delete(x_epochs,m,axis=0)
                y_epochs=np.delete(y_epochs,m)
        
        #print("batch number:"+str(batch_no))
        #print("global batch no:"+str(global_batch))
        batch_no=batch_no+1
        global_batch=global_batch+1
        yield x_epochs, np_utils.to_categorical(y_epochs, n_classes)

def calculate_num_samples(files):
    #calculate the length of .mat files in terms of no.of 30 sec epochs
    total_length_samples=0
    for filename in files:
        data = sio.loadmat(path_to_mat_folder+"/"+filename,struct_as_record=False)
        len_signal=np.ceil(len(data['signals'][0,0].eeg1[0])/(time_period_sample*batch_size))
       # print(float(len(data['signals'][0,0].eeg1[0]))/(time_period_sample))
       # print(np.ceil(len(data['signals'][0,0].eeg1[0])/(time_period_sample)))
        total_length_samples=total_length_samples+len_signal
    return total_length_samples

n_classes=5
#sample frequency=125 Hz and epochs for sleep staging=30 sec, therefore time slice for each epoch=30*125=3750 
time_period_sample=3750 
mat_files=[]
n_channels=2

path_to_mat_folder="D:/DEEPSLEEP/codes/eeg_annotations/"
for i in os.listdir(path_to_mat_folder):
    mat_files.append(i)


model_m = Sequential()
model_m.add(Conv1D(32, 3, activation='relu', input_shape=(time_period_sample,n_channels)))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(64, 3, activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(128, 3, activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Dropout(0.5))
model_m.add(Flatten())
model_m.add(Dense(n_classes, activation='softmax'))
print(model_m.summary())

model_m.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

# Hyper-parameters
batch_size = 128
epochs = 1

#num_training_samples=calculate_num_samples(mat_files[:1120])
#batches_per_epoch_train=np.ceil(num_training_samples/batch_size)
batches_per_epoch_train=9375
#print(batches_per_epoch_train)
#print(num_training_samples)
#print(num_training_samples/batch_size)
#print(np.ceil(num_training_samples/batch_size))
#num_validation_samples=calculate_num_samples(mat_files[1120:1400])
#batches_per_epoch_valid=np.ceil(num_validation_samples/batch_size)
batches_per_epoch_valid=2370
#print(batches_per_epoch_valid)

# Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
model_m.fit_generator(generator=generator(batch_size,mat_files[:1120], time_period_sample, n_classes,n_channels,batches_per_epoch_train),
                      steps_per_epoch=batches_per_epoch_train,
                      epochs=epochs,
                      #verbose=1,
                      validation_data=generator(batch_size,mat_files[1120:1400], time_period_sample, n_classes,n_channels,batches_per_epoch_valid),
                      validation_steps=batches_per_epoch_valid,
                      shuffle=True,
                      workers=1)

'''for i in range(len(mat_files)):    
    data=sio.loadmat(path_to_mat_folder+"/"+mat_files[i],struct_as_record=False)
    eeg1=data['signals'][0,0].eeg1[0]
    eeg2=data['signals'][0,0].eeg2[0]
    y=data['signals'][0,0].annotation[0]
    
    for j in range(len(y)):
        if y[j]==5:
            y[j]=4

    for j in range(0,len(eeg1),time_period_sample):
        eeg1_epoch=eeg1[j:j+time_period_sample]
        eeg2_epoch=eeg2[j:j+time_period_sample]
        #x_epochs[i,:,0]=eeg1_epoch
        #x_epochs[i,:,1]=eeg2_epoch
        x_epochs.append([eeg1,eeg2])
        y_epochs.append(np.unique(y[j:j+time_period_sample]))

x_train=np.asarray(x_epochs).reshape(-1,time_period_sample,2)
y_train=np.asarray(y_epochs)
print(x_train.shape)
print(y_train.shape)
print(np.unique(y_train))

#x_train = x_train.astype("float32")
#y_train = y_train.astype("float32")
y_train = np_utils.to_categorical(y_train, num_classes)

#x_train_instances=x_train.shape[0]
#input_shape1=time_period_sample*1
#x_train=x_train.reshape(x_train_instances,input_shape1)
#print(x_train.shape)

model_m = Sequential()
#model_m.add(Reshape((time_period_sample, 1), input_shape=(input_shape1,)))
model_m.add(Conv1D(32, 3, activation='relu', input_shape=(time_period_sample,2)))
#model_m.add(Conv1D(100, 10, activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(64, 3, activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(128, 3, activation='relu'))
model_m.add(MaxPooling1D(3))
#model_m.add(Conv1D(160, 10, activation='relu'))
#model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.5))
model_m.add(Flatten())
model_m.add(Dense(num_classes, activation='softmax'))
print(model_m.summary())



history = model_m.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      validation_split=0.2,
                      verbose=1)'''