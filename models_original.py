# _*_ coding" utf-8  _*_

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
#import keras

# settings
#epochs=50
epochs=10
batch_size=256

#get data
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, _), (x_test, _) = mnist.load_data()

# adjust minst data
x_train  = x_train.reshape(60000, 784)
x_test   = x_test.reshape(10000, 784)
x_train  = x_train.astype('float32')
x_test   = x_test.astype('float32')
x_train /= 255
x_test  /= 255
#y_train  = keras.utils.to_categorical(y_train, 10)
#y_test   = keras.utils.to_categorical(y_test, 10)

# compile - model
encoding_dim = 32 # num. of dim in mid-layer
input_img = Input(shape=(784,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)
autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# learning
autoencoder.fit(x_train, x_train,
                epochs = epochs,
                batch_size = batch_size,
                shuffle= 'True',
                verbose = 1,
                validation_data=(x_test, x_test)
               )

# check score 
#score = autoencoder.evaluate(x_test, x_test, verbose=1)
#print()
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

autoencoder.save('./autoencoder.h5')

#### plot
import matplotlib.pyplot as plt

# comvert test-images
decoded_imgs = autoencoder.predict(x_test) 

# show
n = 10
plt.figure(figsize=(16,8))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
