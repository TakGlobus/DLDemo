# _*_ coding" utf-8  _*_

"""
  + Descriptoin
      demo code for autoencoder

      data : necp reanaltysis 1.0 by 1.0
            Variable   :  surfae level presure
            Train data : 2018/05/01 00UTC - 2018/07/31 18UTC
            Test  data : 2018/08/01 00UTC - 2018/08/20 18UTC

      result : 
        * Surface Pressure
      100/100epochs ==> loss: 0.3081 - val_loss: 0.2932
        * RH
      100/100epochs ==> loss: 0.6314 - val_loss: 0.6428
        * PWAT: Total Precipitation Water
      100/100epochs ==> loss: 0.5976 - val_loss: 0.6313


  + Hisotry

     ver      date      editor       description
  ----------------------------------------------------------------------
     1.0   Aug.30.18   T.Kurihana   conv. autoencoder model


  + Architecture
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 60, 60, 1)         0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 60, 60, 16)        160
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 30, 30, 16)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 30, 30, 8)         1160
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 15, 15, 8)         0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 15, 15, 8)         584
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 8, 8, 8)           0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 8, 8, 8)           584
_________________________________________________________________
up_sampling2d_1 (UpSampling2 (None, 16, 16, 8)         0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 16, 16, 8)         584
_________________________________________________________________
up_sampling2d_2 (UpSampling2 (None, 32, 32, 8)         0
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 30, 30, 16)        1168
_________________________________________________________________
up_sampling2d_3 (UpSampling2 (None, 60, 60, 16)        0
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 60, 60, 1)         145
=================================================================
Total params: 4,385
Trainable params: 4,385
Non-trainable params: 0
_________________________________________________________________
"""

from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.models import Model
import numpy as np

from mod_gendata import *
from mod_cutregion import *

# usr-settings
keyward='pwat'
#epochs=1000
epochs=100
#epochs=50
#epochs=30
#epochs=10
batch_size=256
inputdir='/home/kurihana/ml_model/work_mymodel/ex4/data/train_data'
testdir='/home/kurihana/ml_model/work_mymodel/ex4/data/test_data'

# original data shape
lon=360
lat=181

# convert/plot-settings
nlat = 60
nlon = 60
n = 2 # number of pics on screen

#get data
gd = gen_grads_data()
x_train  = gd.load_key_data(inputdir, keyward)
x_test   = gd.load_key_data(testdir, keyward)
x_train  = x_train.reshape(x_train.shape[0],lat, lon)
x_test   = x_test.reshape(x_test.shape[0],lat, lon)

# adjust minst data
x_train  = x_train.astype('float32')
x_test   = x_test.astype('float32')

# select region for square shape image
num_of_dim = 3  # dimension of load_data
sg = sl_region()
# north lat, south lat, weat lon, east lon
x_train   = sg.get_region(x_train,3,lat,lon,60,0,225,285 )
x_test    = sg.get_region(x_test,3,lat,lon,60,0,225,285 )
x_train   = x_train.reshape(x_train.shape[0],60, 60,1)
x_test    = x_test.reshape(x_test.shape[0],60, 60,1)


### compile conv-model
input_img = Input(shape=(nlat,nlon,1)) # reshape by addingannel

# Encoding part

encoded = Conv2D(16,(3,3), activation='relu', padding='same')(input_img)
encoded = MaxPooling2D(pool_size=(2,2), padding='same')(encoded)
encoded = Conv2D(8,(3,3), activation='relu', padding='same')(encoded)
encoded = MaxPooling2D(pool_size=(2,2), padding='same')(encoded)
encoded = Conv2D(8,(3,3), activation='relu', padding='same')(encoded)
encoded = MaxPooling2D(pool_size=(2,2), padding='same')(encoded)

# Decoding part

decoded = Conv2D(8,(3,3), activation='relu', padding='same')(encoded)
decoded = UpSampling2D((2,2))(decoded)
decoded = Conv2D(8,(3,3), activation='relu', padding='same')(decoded)
decoded = UpSampling2D((2,2))(decoded)
decoded = Conv2D(16,(3,3), activation='relu')(decoded)
decoded = UpSampling2D((2,2))(decoded)
decoded = Conv2D(1,(3,3), activation='relu', padding='same')(decoded)

#
autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# model summary
print(autoencoder.summary())

# learning
autoencoder.fit(x_train, x_train,
                epochs = epochs,
                batch_size = batch_size,
                shuffle= 'True',
                verbose = 1,
                validation_data=(x_test, x_test)
               )

# save trained model
autoencoder.save('./'+keyward+'_convae'+str(epochs)+'.h5')

#### plot
import matplotlib.pyplot as plt

# comvert test-images
decoded_imgs = autoencoder.predict(x_test) 

# show
plt.figure(figsize=(16,8))
image_list = [0, 12]
#for i in range(n):
for index, i in enumerate(image_list):
    ax = plt.subplot(2,n,index+1)
    plt.imshow(x_test[i].reshape(nlat,nlon))
    plt.gray()
    itime = int(i*6)
    plt.title('+ %d hour'% itime )
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2,n,index+1+n)
    plt.imshow(decoded_imgs[i].reshape(nlat,nlon))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
