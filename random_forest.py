import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from DataImporter import DataImporter
from ActivityDictionary import *
from sklearn import tree
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import graphviz
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRFClassifier
from sklearn.preprocessing import LabelEncoder


TRAIN = [*range(26, 30), 31, 33, 34, *range(39, 45), *range(46, 60)]
TEST = [*range(60, 64)]
LABELS = ["walkfast", "walkmod", "walkslow", "upstairs",
          "downstairs", "lying", "jogging", "sitting", "standing"]


def squeeze_classes(df):
    df.replace("walkfast", "high", inplace=True)
    df.replace("walkmod", "medium", inplace=True)
    df.replace("upstairs", "high", inplace=True)
    df.replace("downstairs", "high", inplace=True)
    df.replace("lying", "low", inplace=True)
    df.replace("jogging", "high", inplace=True)
    df.replace("sitting", "low", inplace=True)
    df.replace("standing", "low", inplace=True)
    df.replace("walkslow", "medium", inplace=True)

    return df


def train_model(df_train, df_test, graph=False):
    feature_names = ["s1", "s2", "s3", "s4", "s5", "s6"]
    # class_names = np.unique(df_train.iloc[:, -1])
    class_names = LABELS
    x_train = df_train.iloc[:, 0:-1]
    x_test = df_test.iloc[:, 0:-1]
    y_train = df_train.iloc[:, -1]
    y_test = df_test.iloc[:, -1]
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    classifier = XGBRFClassifier(
        n_estimators=50, subsample=0.9, colsample_bynode=0.2, objective="binary:logistic", max_depth=20)

   # classifier = RandomForestClassifier(random_state=405, max_depth=15)

    classifier.fit(x_train, y_train)
    y_predict = classifier.predict(x_test)
    # feature_importances = {
    #     'Feature': feature_names,
    #     'Importance': classifier.feature_importances_
    # }

    # feature_importances_df = pd.DataFrame.from_dict(feature_importances)
    # print(feature_importances_df)
    if graph:
        cm = confusion_matrix(
            y_test,
            y_predict,
            labels=classifier.classes_
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=le.classes_
        )
        disp.plot(xticks_rotation=45)
        plt.show()

    return accuracy_score(y_test, y_predict)


def test_for_all():
    di = DataImporter()
    df_test = pd.DataFrame()
    df_train = pd.DataFrame()

    df_train = pd.read_csv("df_train.csv")
   # df_train = squeeze_classes(df_train)

    df_test = pd.read_csv("df_test.csv")
   # df_test = squeeze_classes(df_test)

    acc = train_model(df_train, df_test, graph=False)
    print(acc)


def test_for_one_preson(person, train_test_split=0.2):
    di = DataImporter()
    df = di.get_all_for_person_chunked(person)
    df = squeeze_classes(df)
    df = df.sample(frac=1)
    n_test = math.floor(train_test_split*len(df))
    df_train = df.head(len(df)-n_test)
    df_test = df.tail(n_test)
    acc = train_model(df_train, df_test, graph=True)
    return acc


def test_for_all_people():
    total_acc = 0
    for person in TRAIN:
        print(person)
        total_acc += test_for_one_preson(person)
    for person in TEST:
        print(person)
        total_acc += test_for_one_preson(person)
    total_acc /= (len(TRAIN) + len(TEST))
    return total_acc
