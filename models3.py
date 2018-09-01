# _*_ coding" utf-8  _*_

from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.models import Model
import numpy as np

from mod_gendata import *

# usr-settings
keyward='rh'
#epochs=1000
#epochs=50
epochs=10
batch_size=256
nkernel=2
inputdir='/home/kurihana/ml_model/work_mymodel/ex4/data/train_data'
testdir='/home/kurihana/ml_model/work_mymodel/ex4/data/test_data'

# plot-settings
lon=360
lat=181
n = 2 # number of pics on screen

#get data
gd = gen_grads_data()
x_train  = gd.load_key_data(inputdir, keyward)
x_test   = gd.load_key_data(testdir, keyward)
x_train  = x_train.reshape(x_train.shape[0],lat, lon,1)
x_test   = x_test.reshape(x_test.shape[0],lat, lon,1)

# adjust minst data
x_train  = x_train.astype('float32')
x_test   = x_test.astype('float32')


### compile conv-model
input_img = Input(shape=(lat,lon,1))
# 1
ndim1=lat
encoded=Conv2D(1,3,3, activation='relu')(input_img)
#encoded=MaxPooling2D(pool_size=(2,2))(encoded)
# 2
ndim2=int(ndim1/2)
encoded=Conv2D(8,3,3, activation='relu')(encoded)
#encoded=MaxPooling2D(pool_size=(2,2))(encoded)
# 3
ndim3=int(ndim2/2)
encoded=Conv2D(16,3,3, activation='relu')(encoded)
#encoded=MaxPooling2D(pool_size=(2,2))(encoded)
# 4-back
ndim4=ndim3
decoded=Conv2DTranspose(16,3,3, activation='relu')(encoded)
# 5-stable
decoded=Conv2D(16,3,3, activation='relu') (decoded)
# 6-back
ndim5=ndim2
decoded=Conv2DTranspose(8,3,3, activation='relu')(decoded)
# 7-stable
decoded=Conv2D(8, 3,3, activation='relu')(decoded)
# 8-back
ndim5=ndim2
decoded=Conv2DTranspose(1,3,3, activation='relu')(decoded)
# 9-stable
#decoded=Conv2D(1, 3,3, activation='relu', border_mode='same')(decoded)
#
autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# model summary
print(autoencoder.summary())

#stop
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

autoencoder.save('./'+keyward+'_convae'+str(epochs)+'.h5')

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
