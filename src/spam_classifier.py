# SpamClassifier
#Model for checking if the mail is spam or not

import os
import codecs

from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
import pandas as pd
import numpy as np


DATA_DIR = '../data/spam_2'


def getFeatures(file_data):
    #features = []
    if file_data.find("\n\n"):
        return 1
    else:
        return 0

def scanFiles():
    num =0;
    for (dirpath, dirnames, filenames) in os.walk(DATA_DIR):
        for file in filenames:
            full_path = os.path.join(dirpath, file)
            with codecs.open(full_path, 'r', encoding='"ISO-8859-1') as filehandle:
                file_data = filehandle.read()
                num += getFeatures(file_data)
    print('Total num - ', num)




#Split for validation test set
#At first lets build the simpliest model which just seeks through the text
#for certain words
#Use pipelines

def main():
    print("Starting the classifier...")
    scanFiles()


if __name__ == "__main__":
    main()