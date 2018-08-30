# _*_ coding : utf-8   _*_

import numpy as np
import matplotlib.pyplot as plt
from mod_gendata import *


class sl_region:

  def _init_(self):
      pass

  def get_region(self,
                 load_data,ndim,
                 lat,  lon, 
                 nlat, slat,
                 wlon, elon
                ):
      # Reshape data for optimal shape
      #
      """
      lon : 0  - 360
      lat : 90 - -90
      """
      if ndim == 1: 
          print()
          print( "  Reshape Global Data Time and "+str(lon)+" by "+str(lat)+" "  )
          print( "  if Data shape is different, fix parameters " )
          ntime = int(load_data.shape[0]/(lat*lon))
          load_data = load_data.reshape(ntime, lat, lon)
      if ndim == 2:
          print()
          print( "  Reshape Global Data "+str(lon)+" by "+str(lat)+" "  )
          print( "  if Data shape is different, fix parameters " )
          load_data = load_data.reshape(load_data.shape[0], lat, lon)

      # select region
      #region_data=data[30:90,225:285] # 60 ; 60
      ntime = load_data.shape[0]
      data  = []
      for itime in range(ntime):
        data_array  = load_data[itime][:][:]
        data       += [data_array[90-nlat:90-slat,wlon:elon]] # 60 ; 60
      print(np.asarray(data).shape)
      return np.asarray(data)


# usr-settings
#keyward='sflp'
#inputdir='/home/kurihana/ml_model/work_mymodel/ex4/data/train_data'

# plot-settings
#lon=360
#lat=181

#get data
#gd = gen_grads_data()
#x_train  = gd.load_key_data(inputdir, keyward)
#x_train  = x_train.reshape(x_train.shape[0],lat, lon)

# adjust data shape
#x_train  = x_train.astype('float32')

#data = x_train[0][:][:]
#print(data.shape)

#region_data=data[90:150,225:285] # 60 ; 60
#region_data=data[30:90,225:285] # 60 ; 60
#print(region_data.shape)

#plt.figure(figsize=(16,8))
#plt.imshow(region_data)
#plt.gray()
#plt.show()
