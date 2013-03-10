# encoding: utf-8

import requests as rq
from googlemaps import GoogleMaps

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import uuid

g = GoogleMaps('AIzaSyDvPNDp_QiRGaBPXxaYuY1ska9-uuger8s')

class SmoothRide(object):

    def __init__(self, filenames=None, name=None, notes=None, convert=False,
                 param_list=None):
        self.uuid = uuid.uuid1()

        if name is None:
            self.name = 'SR_object_' + str(self.uuid)
        else:
            self.name = name

        if notes is not None:
            self.notes = notes
        if filenames is not None:
            self.read_data_records(filenames, convert=convert)
        else:
            raise


    def __str__(self):
        self.describe()


    def read_data_records(self, filenames, convert=False,
                          param_list=None):
        """
        Reads data recording from files listed in filenames. Populates
        SmoothRide object with `data` attribute, containing converted
        data set.
        """
        df = pd.DataFrame()
        #for filename in filenames:
        df = pd.read_csv(filenames)
        col_names = pd.Series(df.columns)
        df.index = pd.to_datetime(df.time, errors='raise', utc=True)
        df = df.drop(col_names[col_names.str.contains('[Tt]ime')], axis=1)

        if(convert):
            df = _convert_timestamp(df)
        #df = pd.concat(df)

        self.data = df
                    

    def label_match(self, pat, axis=1):
        if axis == 1:
            s = pd.Series(self.data.columns)
            return self.data[s[s.str.contains(pat)]]
        else:
            s = pd.Series(self.data.index)
            return self.data.ix[s[s.str.contains(pat)]]


    def describe(self):
        print 'SmoothRide Object'
        print 'Name: ', self.name
        print 'UUID: ', self.uuid

        if hasattr(self, 'notes'):
            print 'Notes: ', self.notes
        print '**Time**'
        print '- start:    ', self.data.index[0]
        print '- end:      ', self.data.index[-1]
        print '- duration: ', self.data.index[-1]-self.data.index[0]
        print '**Location**'
        loc_start = self.data[['lat','long']][:1000].mean().round(5).tolist()
        loc_end = self.data[['lat','long']][-1000:].mean().round(5).tolist()
        print '- start:    ', loc_start, \
               ', ', g.latlng_to_address(*loc_start).encode()
        print '- end:      ', loc_end, \
               ', ', g.latlng_to_address(*loc_end).encode()
        print '- distance: ', '<to be implemented>'
        print 'Data Params: ', self.data.columns.tolist()


def _analyze_params(param_list):
    #with open(filename, 'r') as f:
    #    first_line = pd.Series(f.readline().strip().split(','))
    re_str = '([_|.]?)([X|Y|Z])$'
    vector_mask = param_list.str.contains(re_str)
    scalar_mask = ~param_list.str.contains(re_str)
    vector_names = param_list[vector_mask].str.replace(re_str, '').unique()
    scalar_names = param_list[scalar_mask]
    data_params = {'vector': {'mask': vector_mask,
                              'names' : vector_names},
                   'scalar': {'mask': scalar_mask,
                              'names': scalar_names}}
    return data_params
    

def quick_plots(self, c=['g', 'r', 'b'], param_list=None, **kwargs):
    fs = (15,4)
    c = ['g', 'r', 'b']
    plt.figure(figsize=fs)
    for p in param_list:
        self[p].plot(figsize=fs, color=c, **kwargs)


def _convert_timestamp(df):
    start = df.index[0] + pd.tseries.offsets.DateOffset(seconds = 1)
    end = df.index[-1] - pd.tseries.offsets.DateOffset(seconds = 1)
    df = df[start:end]

    #Create timestamp data with sub second resolution
    df['timestamp_mod'] = 1
    for dt in df.index.unique():
        l = len(df.ix[dt])
        if(l == 20):
            dr = pd.date_range(dt, periods=20, freq='50L')
            df.timestamp_mod[dt] = dr
        else:
            start = dt
            f = str(int(1.0*1000000.0/l)) + 'U'
            dr = pd.date_range(start, periods=l, freq=f)
            df.timestamp_mod[dt] = dr

    df.index = pd.to_datetime(df.timestamp_mod)
    df.index.name = 'time'
    df.index = df.index.tz_localize('UTC')
    df = df.drop(['timestamp_mod'], axis=1)

    return df.resample('50L', fill_method='ffill')


'''
def temp_func(param_list):
    #Option to restrict to param_list
    if param_list is not None:
        m = col_names.astype('bool')
        m[:] = False
        for p in param_list:
            m = m | col_names.str.contains(p)
            if m.all() is False:
                print 'File does not contain any of the parameters specified!'
                return 
            df = df[col_names[m]]
            col_names = pd.Series(df.columns)

            data_params = _analyze_params(col_names)
                    
            #Build vector column list
            col_list = []
            vec_names = data_params['vector']['names']
            for vn in vec_names:
                col_list += [vn]*3
                col_list = [col_list, ['X', 'Y', 'Z']*len(vec_names)]

                #Build vector DataFrame
                vec_mask = data_params['vector']['mask']
                df_vec = df[col_names[vec_mask]]
                df_vec.columns = col_list
                return df_vec
'''