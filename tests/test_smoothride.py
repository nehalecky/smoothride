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

_file1_path = os.path.join(curpath(), 'data_recording1.csv')
_notes = """The sample recording data set was generated in a car trip
in Reno, NV."""

_record = sr.FlightRecord(filenames=_file1_path,
                          name='Sample Data for test case.',
                          notes=_notes,
                          convert=True)


class TestFlightRecord(unittest.TestCase):
    def setUp(self):
        self.dirpath = curpath()
        self.csv1 = os.path.join(self.dirpath, 'data_recording.csv')


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
