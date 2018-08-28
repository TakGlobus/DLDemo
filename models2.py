# _*_ coding" utf-8  _*_

from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

from mod_gendata import *

# usr-settings
keyward='rh'
epochs=50
#epochs=10
batch_size=256
inputdir='/home/kurihana/ml_model/work_mymodel/ex4/data/train_data'
testdir='/home/kurihana/ml_model/work_mymodel/ex4/data/test_data'

# plot-settings
lon=360
lat=181
n = 2 # number of pics on screen

#get data
gd = gen_grads_data()
x_train = gd.load_key_data(inputdir, keyward)
x_test  = gd.load_key_data(testdir, keyward)
xdim = x_train.shape[1]
print(x_train.shape[1])
#stop

# adjust minst data
x_train  = x_train.astype('float32')
x_test   = x_test.astype('float32')

# compile - model
encoding_dim = 32 # num. of dim in a mid-layer
input_img = Input(shape=(xdim,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(xdim, activation='sigmoid')(encoded)
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

autoencoder.save('./'+keyward+'_ae50.h5')

#### plot
import matplotlib.pyplot as plt

# comvert test-images
decoded_imgs = autoencoder.predict(x_test) 

# show
plt.figure(figsize=(16,8))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(lat,lon))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[i].reshape(lat,lon))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
