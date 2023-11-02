# This is a sample Python script.
from DataImporter import DataImporter
from ActivityDictionary import *
from PCA_analysis import PCA_analyzer
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    di = DataImporter()
    di.get_data(26, DOWNSTAIRS)
    di.get_data(26, SITTING)
    di.get_data(27, DOWNSTAIRS)
    di.get_data(27, SITTING)

    df = di.get_data(27, SITTING)

    pca = PCA_analyzer(df, 6)
    pca.print_pca_data(df.columns)
