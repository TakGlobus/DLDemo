# _*_ coding : utf-8 _*_

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from mod_gendata import *
from eofs.standard import Eof
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# settinigs
testdir = './data/test_data'
keyward = 'pwat'
lat     = 181
lon     = 360

# load data
gd=gen_grads_data()
test_data = gd.load_key_data(testdir, keyward)
test_data = test_data.astype('float32')

# load model
#autoencoder = load_model('rh_ae50.h5') # 50 epochs
#autoencoder = load_model('pwat_ae50.h5') # 50 epochs
#autoencoder = load_model('rh_ae1000.h5') # 1,000 epochs
autoencoder = load_model('pwat_ae1000.h5') # 1,000 epochs
#
decoded_imgs = autoencoder.predict(test_data)  # prediction

# row dataset
#decoded_imgs = test_data  # row/test data

##### eof
# set up latitude
lat_array = np.asarray([x for x in range(-90, 91)])
cos_lat   = np.cos(np.deg2rad(lat_array))
weighted_lat = np.sqrt(cos_lat)[:,np.newaxis] # give a new axis by np.newaxis

# solver
ntime = decoded_imgs.shape[0]#/(lat*lon)
solver    = Eof(decoded_imgs.reshape(ntime,lat, lon), weights=weighted_lat)

#get analysis
eof1 = solver.eofsAsCorrelation(neofs=1)
pc1  = solver.pcs(npcs=1, pcscaling=1)
var1 = solver.varianceFraction(neigs=1)
print('Variance 1st mode %2.2f' %(var1) )
varall = solver.varianceFraction()
print('Variance all mode', varall )
print("Sum of all mode's variance  = ", sum(varall))

#plotting
lons  = np.asarray([x for x in range(0,lon)])
lats  = lat_array
clevs = np.linspace(-1,1,11)
fig   = plt.figure(figsize=(12,8))
ax    = plt.axes(projection=ccrs.PlateCarree(central_longitude=0))
fill  = ax.contourf(lons, lats, eof1.squeeze(), clevs,
                    transform=ccrs.PlateCarree(), cmap=plt.cm.RdBu_r
                   )
ax.add_feature(cfeature.LAND, facecolor='w', edgecolor='k')
cb    = plt.colorbar(fill, orientation='horizontal')
cb.set_label('Correlation Coefficeint', fontsize=14)
plt.title('EOF1 expressed as Correlation', fontsize=16)

plt.show()
