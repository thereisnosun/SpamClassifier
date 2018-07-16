# SpamClassifier
#Model for checking if the mail is spam or not

import os, re
import codecs

from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
import pandas as pd
import numpy as np


DATA_DIR = '../data/spam_2'


def addSimpleFeature(text_substr, text_body, features):
    if re.search(text_substr, text_body,  re.IGNORECASE):
        features.append(1)
    else:
        features.append(0)


def getFeatures(file_data):
    features = []
    start_index = file_data.find("\n\n")
    if start_index == -1:
        return features

    text_body = file_data[start_index:]

    addSimpleFeature("offer good", text_body, features)
    addSimpleFeature("receive a bonus", text_body, features)
    addSimpleFeature("Your Winnings", text_body, features)
    addSimpleFeature("offer is valid", text_body, features)
    addSimpleFeature("You Won", text_body, features)
    addSimpleFeature("cash", text_body, features)
    addSimpleFeature("GET PAID", text_body, features)
    addSimpleFeature("STOLEN HOME VIDEO", text_body, features)

    expr = re.compile("TAKE \$[^\]]+ OFF")
    if expr.match(text_body.upper()):
        features.append(1)
    else:
        features.append(0)

    if text_body.count('!') > 1:
        features.append(1)
    else:
        features.append(0)

    upper_sum = sum(1 for c in text_body if c.isupper())
    if upper_sum / len(text_body) > 0.5:
        features.append(1)
    else:
        features.append(0)

    return features


def getFeaturesMatrix():
    features_matrix = []
    count = 0
    for (dirpath, dirnames, filenames) in os.walk(DATA_DIR):
        for file in filenames:
            full_path = os.path.join(dirpath, file)
            with codecs.open(full_path, 'r', encoding='"ISO-8859-1') as filehandle:
                file_data = filehandle.read()
                feature_vec = getFeatures(file_data)
                count += 1
                if not feature_vec:
                    continue
                features_matrix.append(feature_vec)

    np_features = np.array(features_matrix)
    print(type(np_features), np_features.shape, type(np_features[0]))
    return np_features



#Split for validation test set
#At first lets build the simpliest model which just seeks through the text
#for certain words
#Use pipelines

def main():
    print("Starting the classifier...")
    features_matrix = getFeaturesMatrix()


if __name__ == "__main__":
    main()