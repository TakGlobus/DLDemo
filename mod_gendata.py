# _*_  coding : utf-8  _*_

import numpy as np
import glob
import copy

class gen_grads_data:

  def _init_(self):
      pass

  def const(self):
      lat = 181
      lon = 360
      return lat, lon

  def get_filelist(self, inputdir):
      return glob.glob(inputdir+'/*')

  def read_binfile(self, inputfile):
      with open(inputfile, 'rb') as ifile:
        data = np.fromfile(ifile, '>f', sep='')
      return data

  def s2n(self, data):
      lat, lon = self.const()
      array2d = data.reshape(lat, lon)
      s2n_array2d = copy.deepcopy(array2d)
      for i in range(lon):
        s2n_array2d[:,i] = array2d[::-1,i]
      return s2n_array2d.flatten()

  def load_data(self, inputdir):
      filelist = self.get_filelist(inputdir)
      # open file
      datalist = []
      for ifile in filelist:
          idata = self.read_binfile(ifile)
          rdata = self.s2n(idata)
          datalist += [rdata]
      # max value for 0-1 normalization
      imax = np.asarray(datalist).max()
      print( ' Max value  ', imax)
      normed_data = np.asarray(datalist)/imax
      #
      return normed_data
