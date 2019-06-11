# -*- coding: utf-8 -*-
"""
Created on Mon May 20 12:09:46 2019

@author: MUJ
"""
import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler
training_sample = []
training_labels = []
for i in range(50):
    rand_num =randint(1,64)
    training_sample.append(rand_num)
    training_labels.append(0)
    
    rand_num =randint(64,100)
    training_sample.append(rand_num)
    training_labels.append(1)
for i in range(1000):
    rand_num =randint(1,64)
    training_sample.append(rand_num)
    training_labels.append(1)
    
    rand_num =randint(64,100)
    training_sample.append(rand_num)
    training_labels.append(0)

training_sample = np.array(training_sample)
training_labels = np.array(training_labels)
scaler = MinMaxScaler()
training_sample = scaler.fit_transform((training_sample).reshape(-1,1))

model = Sequential([
    Dense(16,input_shape = (1,), activation = 'relu'),
    Dense(32,activation = 'relu'),
    Dense(2, activation = 'softmax')
])
model.summary()
model.compile(Adam(lr = .001),loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
model.fit(training_sample,training_labels,validation_split = 0.1,batch_size = 10,epochs = 20, shuffle = 'True',verbose =2)

test_sample = []
test_labels = []
for i in range(50):
    rand_num = randint(1,64)
    test_sample.append(rand_num)
    test_labels.append(1)
    rand_num = randint(64,100)
    test_sample.append(rand_num)
    test_labels.append(0)
    
for i in range(200):
    rand_num = randint(1,64)
    test_sample.append(rand_num)
    test_labels.append(0)
    rand_num = randint(64,100)
    test_sample.append(rand_num)
    test_labels.append(1)
test_sample = np.array(test_sample)
test_labels = np.array(test_labels)
test_sample = scaler.fit_transform((test_sample).reshape(-1,1))
predictions = model.predict(test_sample,batch_size = 10,verbose = 0)
for i in predictions:
    print(i)
predictions_roundoff = model.predict_classes(test_sample,batch_size = 10,verbose = 0)
for i in predictions_roundoff:
    print(i)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

cm = confusion_matrix(predictions_roundoff,test_labels)
def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
plot_confusion_matrix(cm)

model.save("medical_trail_model.h5")
from keras.models import load_model
new_model = load_model("medical_trail_model.h5")
new_model.summary()
new_model.get_weights()

#We use json file to only save the architecture not the weights and training values.

json_string = model.to_json()
from keras.models import model_from_json
model_architecture = model_from_json(json_string)

#if we want to only save the weights then we can use to_save model

weights = model.save_weights("my_model_weights.h5")
model2 = Sequential([
    Dense(16,input_shape = (1,), activation = 'relu'),
    Dense(32,activation = 'relu'),
    Dense(2, activation = 'softmax')
])
model2.load_weights("my_model_weights.h5")