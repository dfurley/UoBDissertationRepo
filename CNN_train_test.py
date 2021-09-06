#script for training and testing CNN

#import libraries
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
from tensorflow.keras.optimizers import SGD,RMSprop,Adam
from keras.utils import np_utils
import pandas as pd
import itertools
import keras

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

#load data
X = pickle.load(open("input_4_X.pickle", "rb"))
y = pickle.load(open("input_4_y.pickle", "rb"))

#split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#normalise data
X_train /= 255.0
X_test /= 255.0

#batch_size to train
batch_size = 16
# number of output classes
nb_classes = 3
# number of epochs to train
nb_epoch = 20

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#configure model
model = Sequential()

model.add(Conv2D(nb_filters, nb_conv, nb_conv,
                        input_shape=X.shape[1:]))
convout1 = Activation('relu')
model.add(convout1)
model.add(Conv2D(nb_filters, nb_conv, nb_conv))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

opt = Adam(learning_rate = 0.0001)
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

#train model
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=270,
                verbose=1, validation_data=(X_test, Y_test))

#plot output
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(270)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#model score
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

#print classification report
predictions = model.predict(X_test)
predictions = np.argmax(predictions,axis=1)
print(classification_report(y_test,predictions,target_names = ['cyanobacteria','diatom','green']))

#plot confusion matrix
#Since our data is in dummy format we put the numpy array into a dataframe and call idxmax axis=1 to return the column
# label of the maximum value thus creating a categorical variable
#Basically, flipping a dummy variable back to it’s categorical variable
categorical_test_labels = pd.DataFrame(y_test)
categorical_preds = pd.DataFrame(predictions)
confusion_matrix_test = confusion_matrix(categorical_test_labels, categorical_preds)

#To get better visual of the confusion matrix:
def plot_confusion_matrix(cm, classes, normalize=False,title ='Confusion Matrix', cmap = plt.cm.Blues):
 
#Add Normalization Option - prints pretty confusion metric with normalization option ‘’’
   if normalize:
     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
     print("Normalized confusion matrix")
   else:
     print('Confusion matrix, without normalization')
 
# print(cm)
 
   plt.imshow(cm, interpolation='nearest', cmap=cmap)
   plt.title(title)
   plt.colorbar()
   tick_marks = np.arange(len(classes))
   plt.xticks(tick_marks, classes, rotation=45)
   plt.yticks(tick_marks, classes)
 
   fmt = '.2f' if normalize else 'd'
   thresh = cm.max() / 2.
   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
 
   plt.tight_layout()
   plt.ylabel('True label')
   plt.xlabel('Predicted label')

plot_confusion_matrix(confusion_matrix_test,['cyanobacteria','diatom','green'],normalize=True)

#validation / unseen data test
#load data
X_unseen = pickle.load(open("test_X.pickle", "rb"))
y_unseen = pickle.load(open("test_y.pickle", "rb"))

#normalise data
X_unseen = X_unseen.astype('float32')
X_unseen /= 255.0

#predictions and classification report
predictions_unseen = model.predict(X_unseen)
predictions_unseen = np.argmax(predictions_unseen,axis=1)
print(classification_report(y_unseen,predictions_unseen,target_names = ['cyanobacteria','diatom','green']))

#prepare confusion matrix for unseen data
categorical_test_labels_unseen = pd.DataFrame(y_unseen)
categorical_preds_unseen = pd.DataFrame(predictions_unseen)
confusion_matrix_unseen = confusion_matrix(categorical_test_labels_unseen, categorical_preds_unseen)

#plot confusion matrix
plot_confusion_matrix(confusion_matrix_unseen,['cyanobacteria','diatom','green'],normalize=True)