# encoding: utf-8
import requests as rq
from pygeocoder import Geocoder

import pymongo
from bson.binary import Binary
from bson.objectid import ObjectId

import tempfile
import os

import uuid
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import json
import pprint as pp

g = Geocoder('AIzaSyDvPNDp_QiRGaBPXxaYuY1ska9-uuger8s') #Google Maps API


class FlightRecord(object):

    def __init__(self, filenames=None, user=None, title=None, notes=None,
                 tags=None, aircraft=None, make_seq=False, set_freq=False,
                 samp_rate=None, param_list=None):
        """
        Initialize FlightRecord object with given paramters, including:

        set_freq : bool, default False
            If True, resamples read data time series to a set frequency. This is
            useful in dealing with the slowdowns in sensor recording rate that
            can occur on mobile devices when sampling at sub second resolution.
        """
        self._samp_rate = samp_rate
        self.user = user
        self.notes = notes
        self.tags = tags
        self.aircraft = aircraft
        self.data = None
        if filenames is not None:
            self.append(filenames, make_seq=make_seq, set_freq=set_freq)


    def _freq(self):
        return _samp_rate_to_freq(self._samp_rate)


    def loc_start(self):
        return self.data[['lat','long']][:1000].mean().round(5).tolist()


    def loc_end(self):
        return self.data[['lat','long']][-1000:].mean().round(5).tolist()


    def time_start(self):
        return self.data.index[0]


    def time_end(self):
        return self.data.index[-1]


    def title(self):
        loc_start = g.reverse_geocode(*self.loc_start()).formatted_address
        title = '{0}: {1} ({2})'
        return title.format(self.user, str(self.time_start()), loc_start)


    def append(self, filenames, make_seq=False, set_freq=False):
        """
        Append new data (or additional data) to FlightRecord data set.
        """
        df_list = []
        if self.data is not None:
            df_list.append(self.data)

        if isinstance(filenames, list):
            for f in filenames:
                df = _read_raw_data(f, make_seq=make_seq, set_freq=set_freq,
                                    samp_rate=self._samp_rate)
                df_list.append(df)
            df = pd.concat(df_list)
            df = df.resample(self._freq())

        else:
            df = _read_raw_data(filenames, make_seq=make_seq, set_freq=set_freq,
                                samp_rate=self._samp_rate)
        #Store in data attribute
        self.data = df


    def __str__(self):
        self.describe()


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
        print 'Title: ', self.title()
        print 'Tags: ', self.tags
        if hasattr(self, 'notes'):
            print 'Notes: ', self.notes
        print '**Aircraft**:', self.aircraft
        print '**Time**'
        print '- start:    ', self.time_start()
        print '- end:      ', self.time_end()
        print '- duration: ', self.time_end() - self.time_start()
        print '**Location**'
        lsname = (g.reverse_geocode(*self.loc_start()).formatted_address
                  .encode('utf8'))
        print ('- start:    {0}, ({1})').format(self.loc_start(), lsname)
        lename = (g.reverse_geocode(*self.loc_end()).formatted_address
                   .encode('utf8'))
        print ('- end:    {0}, ({1})').format(self.loc_end(), lename)
        d = _distance_between_coords(self.loc_start(), self.loc_end())
        print ('- distance, start to end (km): %.2f' % d)
        print '- distance, total        (km): <to be implemented>'
        print 'Data Params: ', self.data.columns.tolist()


    def to_dict_record(self, raw_data=None):
        """
        Casts FlightRecord object to Python dict to allow for database
        insertion.
        """

        record = {'user'       : self.user,
                  'notes'      : self.notes,
                  'time_start' : self.data.index[0],
                  'time_end'   : self.data.index[-1],
                  'loc_start'  : self.loc_start(),
                  'loc_end'    : self.loc_end(),
                  'tags'       : self.tags,
                  'data_params': self.data.columns.tolist(),
                  'raw_data'   : raw_data}
        return record


    def from_dict_record(self, record_dict):
        """
        Casts Python dict (from database query result) to FlightRecord object.
        """
        self.data = _df_from_h5binary(record_dict['raw_data'])
        self.user = record_dict['user']
        self.notes = record_dict['notes']
        self.tags = record_dict['tags']


    def load_rec(self, record):
        """
        Loads record dict into FlightRecord object
        """
        raise NotImplementedError, "To be implemented shortly...."


    def insert_rec(self):
        """
        Persists FlightRecord object (flight record) to remote MongoDB.
        """
        hdf5_binary = _df_to_h5binary(self.data)
        record = self.to_dict_record(raw_data = hdf5_binary)


        #Establish connection
        mongo_user = 'smoothrideclass'
        mongo_pass = 'f4vJCI2RJVQuutL'
        db = Database(mongo_user, mongo_pass)

        #Connect to flights collection and record
        db_collection = 'recordings'
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


def _read_raw_data(filename, make_seq=False, set_freq=False, samp_rate=None,
                   param_list=None):
    """
    Reads raw data recording from files listed in filenames and converts
    to pandas DataFrame. Allows for conversion of timestamp data to set
    frequency.
    """
    df = pd.read_csv(filename)
    col_names = pd.Series(df.columns)
    df.index = pd.to_datetime(df.time, errors='raise', utc=True)
    df = df.drop(col_names[col_names.str.contains('[Tt]ime')], axis=1)

    #If sample rate is not defined, calculate average sampling rate
    #(num samples per second) directly from the DataFrame and use it.
    if samp_rate is None:
        samp_rate = _detect_samp_rate(df)
    mean_freq = _samp_rate_to_freq(samp_rate)
    df.index = df.index.tz_localize('utc')
    if(make_seq):
        df = _sequentialize(df, samp_rate, mean_freq)
    if(set_freq):
        return df.resample(mean_freq, fill_method='ffill')
    return df


def _sequentialize(df, samp_rate, mean_freq):
    """
    Sequentialize the timestamp information in raw sensor feed to contain sub
    second timestamp resolution (not currently contained in data output).
    """
    a_sec = pd.tseries.offsets.DateOffset(seconds = 1)
    start = df.index[0] + a_sec
    end = df.index[-1] - a_sec
    df = df[start:end]

    #Loop through all single second intervals within time series
    df['timestamp_mod'] = 1
    for dt in df.index.unique():
        num_samples_in_sec = len(df.ix[dt])
        #Case where number of samples within this second interval is equal to
        #the requested sensor sampling rate
        if(num_samples_in_sec == samp_rate):
            dr = pd.date_range(dt, periods=samp_rate, freq=mean_freq)
            df.timestamp_mod[dt] = dr
        #Else, case when number of samples within this second interval is NOT
        #equal to requested sensor sampling rate.
        else:
            #Calculate abnormal frequency, in microseconds, of samples within
            #this second interval
            mod_freq = str(int(1.0*1000000.0/num_samples_in_sec)) + 'U'
            dr = pd.date_range(dt, periods=num_samples_in_sec, freq=mod_freq)
            df.timestamp_mod[dt] = dr

    df.index = pd.to_datetime(df.timestamp_mod)
    df.index.name = 'time'
    df = df.drop(['timestamp_mod'], axis=1)

    return df


def _detect_samp_rate(df):
    """
    Calculates mean sampling rate in df directly from DatetimeIndex.
    """
    if df.index.freq is None:
        num_samples_per_sec = [len(df.ix[dt]) for dt in df.index.unique()]
        return  int(np.ceil(np.mean(num_samples_per_sec)))
    else:
        raise


def _samp_rate_to_freq(samp_rate):
    """
    Converts sampling rate (number of samples per second) to a pandas frequency
    in milliseconds.
    """
    return str(int(1/float(samp_rate) * 1000)) + 'L'


def _df_to_h5binary(df):
    """
    Converts DataFrame to HDF5 Binary
    """
    #Read HDF5 binary into 'memory' by creating a temporary file for writing
    #data in 'data' DataFrame to HDF5 and converting to string.
    #(a hack until PyTables gh-123)
    #See: https://github.com/PyTables/PyTables/pull/173
    tf = tempfile.NamedTemporaryFile()
    h5store = pd.HDFStore(tf.name, complevel=9, complib='blosc')
    h5store['data'] = df
    h5store.close()
    fileh = open(tf.name, 'rb')
    hdf5_string = fileh.read()
    fileh.close()
    tf.close()
    h5binary = Binary(hdf5_string)
    return h5binary


def _df_from_h5binary(h5binary):
    """
    Returns DataFrame from HDF5 Binary
    """
    #Read HDF5 binary into 'memory' by creating a temporary file for writing
    #data in 'data' DataFrame to HDF5 and converting to string.
    #(a hack until PyTables gh-123)
    #See: https://github.com/PyTables/PyTables/pull/173
    tf = tempfile.NamedTemporaryFile(delete=False)
    fileh = open(tf.name, 'w')
    fileh.write(h5binary.rstrip())
    fileh.close()
    h5store = pd.HDFStore(tf.name)
    os.remove(tf.name)
    df = h5store['data']
    h5store.close()
    return df


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


class Database(object):
    """
    Instantiates a connection with SmoothRide database allowing for standard
    CRUD (create, read, update and delete) operations.
    """

    def __init__(self, user='testing', pwd='F14273cI886PE5g', subdom='ds051447',
                 account_name = 'smoothride',
                 db_loc = 'mongodb://{0}:{1}@{2}.mongolab.com:51447/{3}',
                 db_name = 'smoothride'):

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


class Collection(object):
    """
    Intantiates a connection to a collection in SmoothRide database.
    """
    def __init__(self, collection='testing', **kwargs):
        db = Database(**kwargs)
        self._coll = db[collection]


    def find(self, query=None, data_projection=False):
        """
        Search the collection for results matching query. Returns a projection
        of the record NOT containing `raw_data` or `data_params`.
        """
        if isinstance(query, ObjectId):
            query = {'_id': query}
        elif isinstance(query, str): # Convert from string to ObjectId:
            query = ObjectId(query)
        if data_projection is True:
            return self._coll.find(query)
        else:
            return self._coll.find(query, {'raw_data':0, 'data_params':0})


    def find_one(self, query=None, data_projection=False):
        """
        Search the collection for a single result  matching query. Returns a projection
        of the record NOT containing `raw_data` or `data_params`.
        """
        if isinstance(query, ObjectId):
            query = {'_id': query}
        elif isinstance(query, str): # Convert from string to ObjectId:
            query = ObjectId(query)
        if data_projection is True:
            return self._coll.find_one(query)
        else:
            return self._coll.find_one(query, {'raw_data':0, 'data_params':0 })


    def list_all(self, query=None):
        """
        Search the collection and list ALL results matching query. (Will
        eventually have to be chunked when db is large).
        """
        cursor = self.find(query)
        for post in cursor:
            print '\n**Record**'
            print pp.pprint(post, indent=1)


    def get_rec(self, record_id):
        """
        Loads a record from SmoothRide database and returns it as a
        FlightRecord object.
        """
        if isinstance(record_id, str): # Convert from string to ObjectId:
            record_id = ObjectId(record_id)
        #Retrieve record
        return self._coll.find_one({'_id': record_id})

        #Convert to FlightRecord Object

#Statistics---------------------------------------------------------------------

def _distance_between_coords(coord1, coord2):
    """
    Calculate the great circle distance (km) between two coordinates on the
    earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lat1, lon1 = map(np.radians, coord1)
    lat2, lon2 = map(np.radians, coord2)
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return np.round(km, 2)