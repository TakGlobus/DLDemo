# _*_ coding : utf-8 _*_

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from mod_gendata import *
from eofs.standard import Eof

# settinigs
testdir = './data/test_data'
keyward = 'rh'
lat     = 181
lon     = 360

# load data
gd=gen_grads_data()
test_data = gd.load_key_data(testdir, keyward)
test_data = test_data.astype('float32')

# load model
autoencoder = load_model('rh_ae50.h5')
decoded_imgs = autoencoder.predict(test_data)

##### eof
# set up
lat_array = np.asarray([x for x in range(-90, 91)])
cos_lat   = np.cos(np.deg2rad(lat_array))
weighted_lat = np.sqrt(cos_lat)

import iris 
from eofs.iris import Eof

read a spatial-temporal field, time must be the first dimension
sst = iris.load_cube('sst_monthly.nc')
print(sst.shape)
stop

for idata in decoded_imgs:
  #solver    = Eof(idata.reshape(lat, lon), weights=weighted_lat)
  solver    = Eof(idata.reshape(1, lat, lon), weights='coslat')

  # 
  eof1 = solver.eofsAsCorrelation(neofs=1)
  pc1 = solver.pcs(npcs=1, pcscaling=1)
  print(eof1, pc1)
