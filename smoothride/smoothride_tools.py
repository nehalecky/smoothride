# encoding: utf-8
import requests as rq
from pygeocoder import Geocoder

import pymongo
from bson.binary import Binary
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import uuid

g = Geocoder('AIzaSyDvPNDp_QiRGaBPXxaYuY1ska9-uuger8s') #Google Maps API

class FlightRecord(object):

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
            self.read_raw_data(filenames, convert=convert)
        else:
            raise


    def __str__(self):
        self.describe()


    def read_raw_data(self, filenames, convert=False,
                      param_list=None):
        """
        Reads raw data recording from files listed in filenames. Populates
        FlightRecord object with `data` attribute, containing converted
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
                    

    def label_match(self, pat, tstart=None, tend=None, axis=1):
        if tstart is not None or tend is not None:
            df = self.data.ix[tstart:tend]
        else:
            df = self.data
        if axis == 1:
            s = pd.Series(df.columns)
            return df[s[s.str.contains(pat, case=False)]]
        else:
            s = pd.Series(df.index)
            return df.ix[s[s.str.contains(pat, case=False)]]


    def describe(self):
        print 'FlightRecord Object'
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
        print '- start:    ', loc_start, #\
               #', ', g.reverse_geocode(*loc_start)
        print '- end:      ', loc_end, #\
               #', ', g.reverse_geocode(*loc_end)
        print '- distance: ', '<to be implemented>'
        print 'Data Params: ', self.data.columns.tolist()


    def insert_rec(self):
        """
        Persists FlightRecord object (flight record) to remote MongoDB.
        """
        #Read HDF5 binary into 'memory' by creating a temporary file for writing
        #data in 'data' DataFrame to HDF5 and converting to string.
        #(a hack until PyTables gh-123)
        #See: https://github.com/PyTables/PyTables/pull/173
        temp = tempfile.NamedTemporaryFile()
        h5store = pd.HDFStore(temp.name, complevel=9, complib='blosc')
        h5store['data'] = self.data
        h5store.close()
        fileh = open(temp.name, 'r')
        hdf5_string = fileh.read()
        fileh.close()
        temp.close()
        hdf5_binary = Binary(hdf5_string)

        #CREATE dict for record (eventually to be moved to 'to_dict'' method)
        record = {'user'       : str(self.uuid),
                  'title'      : self.name,
                  'notes'      : self.notes,
                  'time_start' : self.data.index[0],
                  'time_end'   : self.data.index[-1],
                  'loc_start'  : (self.data[['lat','long']][:1000].mean()
                                 .round(5).tolist()),
                  'loc_end'    : (self.data[['lat','long']][:1000].mean()
                                 .round(5).tolist()),
                  'params'     : self.data.columns.tolist(),
                  'tags'       : ['test', 'car'],
                  'data_params': self.data.columns.tolist(),
                  'raw_data'   : hdf5_binary}

        #Establish connection
        mongo_user = 'smoothrideclass'
        mongo_pass = 'f4vJCI2RJVQuutL'
        db = Database(mongo_user, mongo_pass)

        #Connect to flights collection and record
        db_collection = 'flights'
        record_id = db[db_collection].insert(record)
        self._id = record_id
        return record_id


    def quick_plots(self, tstart=None, tend=None,
                    c=['g', 'r', 'b'], param_list=None, fs=(15,4), **kwargs):

        if param_list is None:
            #param_list = _analyze_params(self.data.columns)
            param_list=['Acceleration', 'Rotation']
        #Vector parameters
        for p in param_list:
            plt.figure(figsize=fs)
            ax = self.label_match(p, tstart, tend).plot(figsize=fs,
                                     color=c, grid=True,
                                     title=p,
                                     alpha=0.5,
                                     **kwargs)
            ax.set_ylabel(p)

        #Scalar
        param_list = ['Speed', 'Alt']
        colors = ['b', 'r']
        for p, c in zip(param_list, colors):
            plt.figure(figsize=fs)
            ax = self.label_match(p, tstart, tend).plot(figsize=fs,
                                                        color=c, grid=True,
                                                        title=p, alpha=0.5,
                                                        lw=3.0, **kwargs)
            ax.set_ylabel(p)

        #Postition
        plt.figure(figsize=(12,12))
        if tstart is not None or tend is not None :
            df = self.data.ix[tstart:tend]
        else:
            df = self.data
        ax = df.plot(x='lat', y='long',
                            lw=5,
                            alpha=0.5,
                            grid=True, title='Position')
        ax.set_ylabel('Latitude')
        ax.set_xlabel('Longitude')


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


class Database(object):
    """
    Instantiates a connection with SmoothRide database allowing for standard
    CRUD (create, read, update and delete) operations.
    """

    def __init__(self, user='testing', pwd='F14273cI886PE5g', subdom='ds051447',
                 account_name = 'smoothride',
                 db_loc = 'mongodb://{0}:{1}@{2}.mongolab.com:51447/{3}',
                 db_name = 'smoothride',
                 collection=None):

        db_uri = db_loc.format(user, pwd, subdom, db_name)

        self.name = db_name
        self.loc = db_loc
        self.subdom = subdom
        self.user = user

        try:
            self._conn = pymongo.MongoClient(db_uri)
            self._db = self._conn[db_name]
        except:
            raise


    def __getitem__(self, collection):
        return self._db[collection]


    def __str__(self):
        #is_connected = self._conn.alive()
        output_str = ('DB Connection?: ' + str(self._conn.alive()) + '\n' +
                      'DB Details:     ' + str(self._db) + '\n' +
                      'DB Collections: ' + str(self._db.collection_names()))
        return output_str


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