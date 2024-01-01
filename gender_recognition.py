#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 04:18:20 2018

@author: sadievrenseker
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense

# ilkleme
classifier = Sequential()

# Adım 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# Adım 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# 2. convolution katmanı
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adım 3 - Flattening
classifier.add(Flatten())

# Adım 4 - YSA
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# CNN ve resimler

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

training_set = train_datagen.flow_from_directory('veriler/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=1,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('veriler/test_set',
                                            target_size=(64, 64),
                                            batch_size=1,
                                            class_mode='binary')

classifier.fit(training_set,
               steps_per_epoch=len(training_set),
               epochs=1,
               validation_data=test_set,
               validation_steps=len(test_set))

import numpy as np
import pandas as pd

test_set.reset()
pred = classifier.predict(test_set, verbose=1)
pred[pred > 0.5] = 1
pred[pred <= 0.5] = 0

print('prediction gecti')

test_labels = []

for i in range(0, int(len(test_set))):
    test_labels.extend(np.array(test_set[i][1]))

print('test_labels')
print(test_labels)

dosyaisimleri = test_set.filenames

sonuc = pd.DataFrame()
sonuc['dosyaisimleri'] = dosyaisimleri
sonuc['tahminler'] = pred.flatten()
sonuc['test'] = test_labels

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, pred)
print(cm)
