import sys
#assuming smoothride_tools.py is located in directory above
#Later, implement more robust method described here
#http://stackoverflow.com/questions/279237/python-import-a-module-from-a-folder
sys.path.append('../smoothride/')

import smoothride_tools as sr

import os

import nose
import unittest

import numpy as np
import pandas as pd


def curpath():
    pth, _ = os.path.split(os.path.abspath(__file__))
    return pth

_file1_path = os.path.join(curpath(), 'data_record_small.csv')
#_record = sr.FlightRecord(filenames=_file1_path,
#                          notes='Sample small data for test case.',
#                          set_freq=True)
_df = pd.read_csv(_file1_path)
_df.index = pd.to_datetime(_df.time, errors='raise', utc=True)
_col_names = pd.Series(_df.columns)
_df = _df.drop(_col_names[_col_names.str.contains('[Tt]ime')], axis=1)

class TestCoreMethods(unittest.TestCase):


    def test_samp_rate_to_freq(self):
        expected = '50L'
        result = sr._samp_rate_to_freq(20)
        self.assertEqual(result, expected)

        expected = '2L'
        result = sr._samp_rate_to_freq(500)
        self.assertEqual(result, expected)


    def test_detect_samp_rate(self):
        expected = 20
        result = sr._detect_samp_rate(_df)
        self.assertEqual(result, expected)


    def test_sequentialize(self):
        df = sr._sequentialize(_df, 20, '50L')
        result = str(df.index[-1] - df.index[0])
        expected = '0:00:22.950000'
        self.assertEqual(result, expected)

        isSeries = isinstance(_df.ix['2013-02-18 18:24:14'], pd.Series)
        self.assertFalse(isSeries)

        isSeries = isinstance(df.ix['2013-02-18 18:24:14'], pd.Series)
        self.assertTrue(isSeries)

        expected = {'RotationX': -0.021989990000000001,
                    'RotationY': 0.13913789999999998,
                    'RotationZ': -0.031529429999999997,
                    'accelerationX': -0.11575319999999999,
                    'accelerationY': -0.053253170000000002,
                    'accelerationZ': -1.015533,
                    'alt': 1466.0,
                    'course': 76.289059999999992,
                    'horizontalAccuracy': 5.0,
                    'lat': 39.51567,
                    'long': -119.8884,
                    'speed': 28.539179999999998,
                    'verticalAccuracy': 6.0}
        result = df.ix['2013-02-18 18:24:01'].to_dict()
        self.assertEqual(result, expected)


    def test_read_raw_data(self):
        df = sr._read_raw_data(_file1_path)
        self.assertEqual(df.index.freq, None)
        result = (df.index[-1] - df.index[0]).total_seconds()
        expected = 25.0
        self.assertEqual(result, expected)


class TestFlightRecord(unittest.TestCase):
    def setUp(self):
        self.dirpath = curpath()
        self.csv1 = os.path.join(self.dirpath, 'data_recording.csv')
    """
    def test_read_raw_data(self):
        flight = _record
        cols = ['lat', 'long', 'alt', 'speed', 'course',
                'verticalAccuracy', 'horizontalAccuracy',
                'accelerationX', 'accelerationY', 'accelerationZ',
                'RotationX', 'RotationY','RotationZ']

        self.assertEqual(flight.data.columns.tolist(), cols)

        result = [4.255704697986578, 5.231543624161074]
        expected = flight.label_match('accuracy').mean().tolist()
        self.assertEqual(result, expected)

        self.assertEqual(flight.data.index.inferred_freq, '50L')
    """

    #def test_to_dict():
    #    pass
    #def test_from_dict():
    #    pass


class TestDatabase(unittest.TestCase):
    pass



if __name__ == '__main__':
    # unittest.main()
    # nose.runmodule(argv=[__file__,'-vvs','-x', '--pdb-failure'],
    #                exit=False)
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
