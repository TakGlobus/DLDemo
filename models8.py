# _*_ coding" utf-8  _*_

from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.models import Model
import numpy as np

from mod_gendata import *
from mod_cutregion import *

# usr-settings
keyward='pwat'
#epochs=1000
#epochs=100
epochs=50
#epochs=30
#epochs=10
batch_size=256
#inputdir='/home/kurihana/ml_model/work_mymodel/ex4/data/train_data'
inputdir='/home/kurihana/ml_model/work_mymodel/ex4/data/train_aug_data'
testdir='/home/kurihana/ml_model/work_mymodel/ex4/data/test_data'

# original data shape
lon=360
lat=181

# convert/plot-settings
nlat = 64
nlon = 64
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
x_train   = sg.get_region(x_train,3,lat,lon,64,0,225,289 )
x_test    = sg.get_region(x_test,3,lat,lon,64,0,225,289 )
#x_train   = sg.get_region(x_train,3,lat,lon,90,-38,200,328 )
#x_test    = sg.get_region(x_test,3,lat,lon,90,-38,200,328 )
x_train   = x_train.reshape(x_train.shape[0],nlat, nlon,1)
x_test    = x_test.reshape(x_test.shape[0],nlat, nlon,1)


### compile conv-model
input_img = Input(shape=(nlat,nlon,1)) # reshape by addingannel

# Encoding part

encoded = Conv2D(4,(2,2), activation='relu', padding='same')(input_img)
encoded = Conv2D(8,(2,2), activation='relu', subsample=(2,2))(input_img)
encoded = Conv2D(16,(2,2), activation='relu', subsample=(2,2))(encoded)
encoded = Conv2D(32,(2,2), activation='relu', subsample=(2,2))(encoded)
encoded = Conv2D(64,(2,2), activation='relu', padding='same')(encoded)

# Decoding part

decoded = Conv2DTranspose(32,(2,2), activation='relu', subsample=(2,2))(encoded)
decoded = Conv2D(16,(2,2), activation='relu', padding='same')(decoded)
decoded = Conv2DTranspose(8,(2,2), activation='relu', subsample=(2,2))(decoded)
decoded = Conv2D(8,(2,2), activation='relu', padding='same')(decoded)
decoded = Conv2DTranspose(4,(2,2), activation='relu', subsample=(2,2))(decoded)
decoded = Conv2D(1,(2,2), activation='relu', padding='same')(decoded)
#decoded = Conv2D(1,(2,2), activation='sigmoid', padding='same')(decoded)

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

# save trained model
autoencoder.save('./'+keyward+'_convae'+str(epochs)+'_2knldeeper_peraug.h5')

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
