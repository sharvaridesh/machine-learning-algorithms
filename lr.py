#!/usr/bin/env python
# File: lr.py
# Author: Sharvari Deshpande <shdeshpa@ncsu.edu>

import os
import struct
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

def read(dataset , path):
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise (ValueError, "dataset must be 'testing' or 'training'")

    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
    get_img = lambda idx: (lbl[idx], img[idx])

    for i in range(len(lbl)):
        yield get_img(i)

def main():
    path = "D:/NCSU/Year 2/Neural Networks"
    dataset = "training"
    train_data = list(read(dataset, path))
    labels = []
    images = []
    for i in range(60000):
        label, pixels = train_data[i]
        pixels = np.array(pixels)
        pixels_flatten = pixels.flatten()
        labels.append(label)
        images.append(pixels_flatten)

    binarizer = preprocessing.Binarizer ()
    X_binarized = binarizer.transform(images)
    X_binarized = pd.DataFrame(X_binarized)
    seed = 42
    clf = LogisticRegression(solver='lbfgs')
    clf.fit (X_binarized, labels)
    score = cross_val_score (clf, images, labels, scoring='accuracy').mean ()
    print (score)

    dataset = "testing"
    test_data = list(read(dataset, path))
    labels = []
    images = []
    for i in range(10000):
        label, pixels = test_data[i]
        pixels = np.array(pixels)
        pixels_flatten = pixels.flatten()
        labels.append(label)
        images.append(pixels_flatten)

    X_binarized = binarizer.transform(images)
    results = clf.predict(X_binarized[:])
    score = clf.score (X_binarized, labels)
    print (score)
    s = pd.Series(results)
    p = pd.get_dummies(s)
    df = pd.DataFrame(p)
    df.to_csv('lr.csv',header=False,index=False)

if __name__ == '__main__':
    main()
