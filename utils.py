import os
import torch

import pandas as pd
import numpy as np


def split_data():
    data = pd.read_csv("dataset/data/label.csv")
    labels = data.groupby("category")
    np.random.seed(10)
    for i, category in enumerate(data.category.unique()):
        randomList = list(range(len(labels.get_group(category))))
        np.random.shuffle(randomList)
        trainPart = labels.get_group(category).iloc[randomList][:int(len(randomList) * 0.8)]
        testPart = labels.get_group(category).iloc[randomList][int(len(randomList) * 0.8):]
        if (i == 0):
            trainAll = trainPart
            testAll = testPart
        else:
            trainAll = pd.concat([trainAll, trainPart], axis=0)
            testAll = pd.concat([testAll, testPart], axis=0)
    trainAll.to_csv("dataset/train.csv", index=False)
    testAll.to_csv("dataset/valid.csv", index=False)


def saveModel(net, path):
    torch.save(net.state_dict(), path)


def data_append(filename, category, dataframe):
    dataframe = dataframe.append({"filename": filename, "category": category}, ignore_index=True)
    return dataframe


def df_create():
    df = pd.DataFrame(columns=['filename', 'category'])
    return df


def csv_save(outputpath, dataframe):
    dataframe.to_csv(outputpath, index=False)


df = df_create()
df = data_append("asd.jpg", 5, df)
df = data_append("bsd.jpg", 4, df)
csv_save('output.csv', df)
