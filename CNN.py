import tensorflow as tf
import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import adam

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


num_classes = 10
class_names =['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)

def cnn():
    model = Sequential()
    
    # Adding more layers to improve the model
    model.add(Conv2D(32, (3, 3),activation='relu', padding = 'same', input_shape=X_train.shape[1:]))
    model.add(Conv2D(32, (3, 3),activation='relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size= (2,2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, (3, 3),activation='relu', padding = 'same'))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size= (2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3),activation='relu', padding = 'same'))
    model.add(Conv2D(128, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size= (2, 2)))
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes,activation='softmax'))
    model.compile(loss= 'categorical_crossentropy', optimizer = 'adam', metrics= ['accuracy'])
    return model

model = cnn()
model.summary()


batch_size = 128
epochs = 20

hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=batch_size)   

print(hist.history.keys())

score = model.evaluate(X_test, Y_test, verbose = 0)
print("Test Loss", score[0])
print("Test accuracy", score[1])

plt.figure(1)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title("Model Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train','validation'], loc = 'upper left')
plt.show()

plt.figure(2)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("Model loss")
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(['train','validation'], loc = 'upper right')
plt.show()

