# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:29:16 2016

@author: grundmann
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from scipy.optimize import leastsq
from scipy.special import  binom
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import sys
import pdb

plt.rcParams[u'keymap.yscale'] = [u'y']
plt.rcParams[u'keymap.fullscreen'] = [u'ctrl+f']
plt.rcParams[u'keymap.save'] = [u'ctrl+s']

def is_number(var_to_check):
    """Checks whether the argument can be converted to a number
    """
    try:
        float(var_to_check)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(var_to_check)
        return True
    except (TypeError, ValueError):
        pass

    return False

def edgedetector(x_array):
    """gives the indices at which the slope changes by comparing the difference for 10
    elements left and right to each other"""
    diff = x_array[1:] - x_array[:-1]
    values = []
    for index, _ in enumerate(diff[10:-10]):
        if index > 20:
            chunk_before = diff[index    : index +  10]
            chunk_after = diff[index + 10: index + 20]
            if all(all(element > chunk_before) for element in chunk_after):
                values.append([x_array[index + 10], '+']) #the rate goes up
            elif all(all(element < chunk_before) for element in chunk_after):
                values.append([x_array[index + 10], '-']) #the rate goes down
    return values

def read_custom_data(path):
    with open(path) as datei:
        content = datei.readlines()
    data_header = content[1].strip('#\n').split(',')
    units = content[2].strip('#\n').split(',')
    units = dict([data_header[index], unit] for index, unit in enumerate(units))
    data = []
    for index, line in enumerate(content[3:]):
        if '#RAWDATA' in line:
            raw_index = index
            break
        else:
            data.append(map(float, line.strip().split(',')))
    data = np.array(data)
    data_frame = pd.DataFrame(data)
    data_frame.units = units
    data_frame.columns = data_header
    try:
        raw_data = []
        raw_header = content[raw_index + 4].strip('#\n').split(',')
        raw_data = []
        for index, line in enumerate(content[raw_index + 5:]):
            raw_data.append(map(float, line.strip().split(',')))
        raw_data = np.array(raw_data)
        data_frame.raw_data = raw_data
        data_frame.raw_header = raw_header
    except:
        pass
    return data_frame

def read_from_mpms_data(path):
    """read data from the mpms into a data frame and return it"""
    if path.split('.')[-1].lower() == 'dat':
        data_frame = pd.read_csv(path, skiprows=30, header=0, usecols=(0, 2, 3, 4))
        data_frame[2] *= 1e-3 #bringing the magnetic moment to SI-units
    elif path.split('.')[-1].lower() == 'raw':
        #first check if the .diag-file does exist in the folder
        #refit
        data, raw_data = raw_reader(path)
        data_frame = pd.DataFrame(data)
        data_frame.raw_data = raw_data
        data_frame.raw_header = ['time in s', 'field in Oe',\
                                 'starting temperature in K',\
                                 'end temperature in K', 'scan number',\
                                 'z-position in cm', 'measured voltage in V',\
                                 'sensitivity factor', 'measurement number']

    data_frame.columns = pd.Index(['time', 'magnetic field', 'temperature', 'magnetic moment'])
    data_frame['magnetic field'] *= 1e-4 #bringing the field to SI-units
    data_frame.units = {'time':'s', 'magnetic field':'T',\
                        'temperature':'K', 'magnetic moment':'J/T'}
    return data_frame

def read_from_nijmegen_data(path):
    """read data from nijmegen into a data frame and return it"""
    with open(path) as datei:
        line = datei.readline()
    if all(is_number(x) for x in line.split()):
        """we calculate the magnetic moment from the capacitance C based on the force F acting
        on the plate due to the magnetic moment m and the field B: F = -m * grad B ~ m * B.
        To first approximation, this force leads to a distance delta x change for the mounting
        plate according to F ~ delta x. This leads to a change of the capacitance C as C ~ 1/x
        for a plate capacitor. Thus, we have m ~ (1/C - 1/C0) / B
        """
        data_frame = pd.read_csv(path, usecols=(1, 6), delimiter='\t')
        data_frame.columns = pd.Index(['magnetic moment', 'magnetic field'])
        #measurements for fields smaller than 0.25T in Nijmegen are
        #generally not giving good data
        data_frame = data_frame[data_frame['magnetic field'] > 0.25]
        #invert measured capacitance to get the plate distance
        data_frame['magnetic moment'] = 1 / data_frame['magnetic moment']
        #find and subract the zero field plate distance
        min_ind = data_frame['magnetic field'].argmin()
        data_frame['magnetic moment'] -= data_frame['magnetic moment'][min_ind]
        #devide by the magnetic field
        data_frame['magnetic moment'] /= np.maximum(data_frame['magnetic field'], 0.01)
        #taking the absolute value
        data_frame['magnetic moment'] = data_frame['magnetic moment'].abs()
        ##        data_frame.units = ['T', 'K', 'arbitrary']
        a, b, g = 0.88468, 5.1038E-6, -2.0107E-9
        pos = 10 * float(path.split()[0])
        T = float(path.split()[0].strip('K'))
        par = [7.96312, -0.00027, -9.5654E-8, 1.4304E-11, -5.7695E-16]
        data_frame['magnetic field'] = a*data_frame['magnetic field']    \
                             + b*data_frame['magnetic field']**3 \
                             + g*data_frame['magnetic field']**5
        faktor = par[0] + pos**2 * par[1] + pos**4 * par[2]\
                        + pos**6 * par[3] + pos**8 * par[4]
        faktor /= par[0]
        data_frame['magnetic field'] = data_frame['magnetic field'] * faktor
#        gradient = 2*pos*par[1] + 4*pos**3*par[2] + 6*pos**5*par[3] + 8*pos**7*par[4]
        data_frame['temperature'] = T
    else:
        data_frame = pd.read_csv(path, skiprows=3, usecols=(1, 2, 3))
        data_frame.columns = pd.Index(['magnetic field', 'temperature', 'magnetic moment'])
        #measurements for fields smaller than 0.25T in Nijmegen are generally not
        #giving sensible information
        #generally not giving good data
        data_frame = data_frame[data_frame['magnetic field'] > 0.25]
        #invert measured capacitance to get the plate distance
        data_frame['magnetic moment'] = 1 / data_frame['magnetic moment']
        #find and subract the zero field plate distance
        min_ind = data_frame['magnetic field'].argmin()
        data_frame['magnetic moment'] -= data_frame['magnetic moment'][min_ind]
        #devide by the magnetic field
        data_frame['magnetic moment'] /= np.maximum(data_frame['magnetic field'], 0.01)
        #taking the absolute value
        data_frame['magnetic moment'] = data_frame['magnetic moment'].abs()

    data_frame['time'] = 0 #add a dummy time variable
    data_frame.units = {'time':'s', 'magnetic field':'T',
                        'temperature':'K', 'magnetic moment':'arbitrary'}
    return data_frame

def readdata(path, sourcetype='MPMS'):
    """Function to read magnetization data from various sources into a data frame
    and add certain analysis functions to the data frame
    """
    import types
    if sourcetype == 'MPMS':
        data_frame = read_from_mpms_data(path)
    elif sourcetype == 'nijmegen':
        data_frame = read_from_nijmegen_data(path)
    elif sourcetype == 'custom':
        data_frame = read_custom_data(path)

    data_frame.derivative = types.MethodType(derivative, data_frame)
    data_frame.smooth = types.MethodType(smooth, data_frame)
    data_frame.binning = types.MethodType(binning, data_frame)
    data_frame.checkfit = types.MethodType(checkfit, data_frame)
    data_frame.path = path
    return data_frame

def smooth(self, x_title='magnetic field', y_title='magnetic moment',
           result_title=None, sigma=1):
    """smoothes the given y_title -data and writes it back to result_title
    (by default overwriting)
    """
    if result_title is None:
        result_title = y_title
    #if X = none, all points are assumed spaced evenly with spacing
    x = self[x_title][self[x_title].notnull() & self[y_title].notnull()]
    y = self[y_title][self[x_title].notnull() & self[y_title].notnull()]
    y_neu = zeros(y.shape[0])
#    pdb.set_trace()
    weights = repeat(x.values, x.shape[0])
    weights = weights.reshape(x.shape[0], x.shape[0])
    weights = x.values - weights
    weights = np.exp(-.5 / sigma**2 * weights ** 2)
    y_neu = (y.values * weights).sum(axis=1) / weights.sum(axis=1)
#    for i in range(y_neu.shape[0]):
#        weights = exp(-.5/sigma**2 * (x[i] - x[:])**2)
#        weights[isnan(weights)] = 0
#    y_neu[i] = dot( y[:].T , weights ) / sum(weights)
    self[result_title][self[x_title].notnull() & self[y_title].notnull()] = y_neu

def binning(self, x='temperature', dimension=1):
    """
    Binning of the data according to one of the input columns. The function tries
    to find the indended spacing (without noise) and prepare corresponding new
    points. We assume symmetric noise for the standard, one-dimensional case,
    we assume equal spacing in 1-dim gpind over the fulle range for the 2-dim
    case, we assume a grid that does not cover the full range.
    """
    from sklearn.cluster import KMeans
    #calculating the differences
    trange = np.arange(self[x].min(), self[x].max(), .1)
    #calculate a density estimate with a gaussian kernel of h=.1, which seems
    #appropriate for usual SQUID-data
    dens = gauss_dens(trange, self[x], h=.1)
    #finding the localmaxima of the estimated density
    ext_indices = argrelextrema(dens, np.greater, mode='wrap')[0]
    #finding the initial centers as the above found extremal points
    centers = trange[ext_indices].reshape(ext_indices.shape[0], 1)
    #feeding the temperature values to the clusterer
    centerer = KMeans(init=centers)
    centerer.fit(self[x].reshape(self[x].shape[0], 1))
    #the found centers replace the initial centers
    centers = Centerer.cluster_centers_
    new_data = np.zeros((centers.shape[0], self.shape[1]))
    for i in range(new_data.shape[0]):
        for index, col in enumerate(self.columns):
            value = self[col][Centerer.labels_ == i].mean()
            self[col][Centerer.labels_ == i] = value
    self.drop_duplicates(inplace=True)

def gauss_dens(X, Y, h):
    """ X: range of values
        Y: data points
    """
    delta = Y[:, np.newaxis] - X
    delta = np.exp(-delta**2/h)
    values = delta.sum(axis=0)
#    for index, element in enumerate(X):
#        values[index] = np.exp(-(element - Y)**2/h).sum()
    return values / np.sqrt(2*np.pi*h) / Y.shape[0]

def derivative(self, x_title='magnetic field', y_title='magnetic moment',
               order=1, method='HBDIFF7', result_title='dm/dB',
               cleanup=True):
    """calculates the derivative and adds it as another coolumns"""
    if not x_title in self.columns:
        raise Exception('The given x_title does not exist i your data.')
    elif not y_title in self.columns:
        raise Exception('The given y_title does not exist i your data.')
    elif x_title == y_title:
        raise Exception('x_title and y_title should not be the same.')
    else:
        x = np.array(self[x_title])
        y = np.array(self[y_title])
        #if the holoborodko scheme is to be used
        if 'HBDIFF' in method:
            N = int(method.strip('HBDIFF'))
            #if N is not odd, we raise it by one
            if not N % 2:
                N += 1
            if N < 5:
                N = 5
            if N < 5:
                raise Exception('The Holoborodko differentiation scheme needs at least N=5.')
            if order == 1:
                z = holo_diff_first_order(x, y, N)
            elif order == 2:
                z = holo_diff_second_order(x, y, N)
            else:
                raise Exception('Only first or second order differentiation available\
                for this method. You asked for order of {0:g}.'.format(order))

    self[result_title] = z

def holo_diff_first_order(x, y, N):
    from numpy import zeros
    M = (N-1) / 2
    m = (N-3) / 2
    out = zeros(x.shape[0])
    L = len(x)

    for k in np.arange(1, M+1, 1):
        zaehler = y[M + k:L - M + k] - y[M - k:L - M - k]
        nenner = x[M + k:L - M + k] - x[M - k:L - M - k]
        nenner[np.where(nenner == 0)] = 1e-3    #to avoid trouble when x is constant
        c = (binom(2 * m, m - k + 1) - binom(2 * m, m - k - 1)) / 2**(2 * m + 1)
        out[M:-M] = out[M:-M] + 2 * k * c * zaehler / nenner
    return out

def __s(k, N):
    M = (N-1) / 2
    if k > M:
        return 0
    elif k == M:
        return 1
    else:
        return ((2 * N - 10) * __s(k + 1, N) - (N + 2 * k + 3) * __s(k + 2, N)) / (N - 2 * k - 1)

def holo_diff_second_order(x, y, N):
    M = (N-1)/2
    L = len(x)
    out = zeros(x.shape[0])
#    a_sum = zeros(L-N+1)
    for k in np.arange(1, M + 1, 1):
        zaehler = y[M+k:L-M+k] + y[M-k:L-M-k] - 2*y[M:L-M]
        nenner = x[M+k:L-M+k] - x[M-k:L-M-k]
        nenner[np.where(nenner == 0)] = 1e-4    #to avoid trouble when the field is constant
        out[M:-M] += 4 * k**2 * __s(k, N) * zaehler / nenner**2
    return out / 2**(N-3)

def central_finite_diff(x, y, N, order):
    #check the length of the input
    L = len(y)
    h = float((max(x) - min(x))/(L - 1))
    out = zeros(L-N+1)
    diffparameters = [[[0, 0, 0, -.5, 0, .5, 0, 0, 0],
                       [0, 0, 1/12., -2/3., 0, 2/3., -1/12., 0, 0],
                       [0, -1/60., 3/20., -3/4., 0, 3/4., -3/20., 1/60., 0],
                       [1/280., -4/105., 1/5., -4/5., 0, 4/5., -1/5., 4/105., -1/280.]],
                      [[0, 0, 0, 1, -2, 1, 0, 0, 0],
                       [0, 0, -1/12., 4/3., -5/2., 4/3., -1/12., 0, 0],
                       [0, 1/90., -3/20., 3/2., -49/18., 3/2., -3/20., 1/90., 0],
                       [-1/560., 8/315., -1/5., 8/5., -205/72., 8/5., -1/5., 8/315., -1/560.]]]

    factors = array(diffparameters[order - 1][(N - 3) / 2])
    for k, factor in enumerate(factors[4-(N-1)/2:5+(N-1)/2]):
        out += factor * y[k : L - N + k + 1]
    return out / h**order

def checkfit(self, abscissa = 'temperature'):
    if not hasattr(self, 'raw_data'):
        print('No raw data provided')
        return
    self.c = chooseablepoints(self , abscissa)

execfile('maganalyzer_functions.py')

if __name__ == "__main__":
#    ar = readdata('20160208_TlCuCl3_M(T,H)_1.8K-3.6K_5.8T-7T.rso.raw')
    ar = readdata('mpms-testdata.rso.analyzed.data', sourcetype='custom')
#    ar = readdata('mpms-testdata.rso.raw')
    ar.checkfit(abscissa='temperature')
