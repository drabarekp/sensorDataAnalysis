# This is a sample Python script.
from DataImporter import DataImporter
from ActivityDictionary import *

if __name__ == '__main__':
    di = DataImporter()
    di.get_data_and_print(26, DOWNSTAIRS)
    di.get_data_and_print(26, SITTING)
    di.get_data_and_print(27, DOWNSTAIRS)
    di.get_data_and_print(27, SITTING)
