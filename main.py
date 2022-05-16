# -*- coding:utf-8 -*-
# 生成特征向量并保存 svm 训练模型
import os, sys
import time
import numpy as np
import json
import xlrd
import requests
import jieba.posseg as pseg
import numpy
from tqdm import tqdm
from sklearn import svm
from sklearn.model_selection import train_test_split
# from sklearn.externals import joblib
from sklearn.metrics import classification_report

curdir = os.path.dirname(os.path.abspath(__file__))
rootdir = os.path.join(curdir, os.path.pardir)
sys.path.append(rootdir)
import pandas as pd
from CW.app.common.similarity import compare

fetures_word = []
with open("data/fetures_oyp.txt") as readme:
    for w in readme.readlines():
        fetures_word.append(w.replace("\n", ""))

y_mark = []
y_vector = []


def load_vector(file_name, ngram=2, mark_one=['1'], mark_two=['-1']):
    global y_mark
    global y_vector

    def similarity(s1, s2):
        try:
            return max(compare(s1, s2, seg=True), compare(s2, s1, seg=True))
        except:
            return 0.0

    num_obs = 1000
    with open(file_name) as f:
        pk = 0
        nk = 0
        for l in tqdm(f.readlines()[:2]):
            if (pk > num_obs) & (nk > num_obs):
                break
            x = l.split("\t")[0]
            # y = l.split("\t")[1]
            try:
                title = l.split("\t")[1]
                post = l.split("\t")[2]
            except:
                title = ""
                post = l.split("\t")[1]
            assert x in ('0', '1', '-1'), "x should be -1 0 1"
            if x in mark_one:
                pk = pk + 1
                if pk <= num_obs:
                    y_mark.append(0)
                else:
                    continue
            elif x in mark_two:
                nk = nk + 1
                if nk <= num_obs:
                    y_mark.append(1)
                else:
                    continue
            else:
                continue
                # raise BaseException("invalide mark")

            y = title + post
            y = pseg.cut(y)
            y = [o.word for o in y if o.flag.startswith("v") or o.flag == "n"]
            vector = []
            for w in fetures_word:
                max_score = 0.0
                for i in range(len(y) - ngram + 1):
                    word = "".join(y[i:i + ngram])
                    max_score = max(similarity(w, word), max_score)
                if max_score < 0.2:
                    max_score = 0.0
                elif max_score > 0.8:
                    max_score = 1.0
                vector.append(max_score)
            y_vector.append(vector)
            print(pk, nk, '\n')


if __name__ == "__main__":
    canshu = {"mark_one": ["1"], "mark_two": ["-1"]}
    start1 = time.time()
    load_vector("data/train.raw.data", **canshu)
    assert len(y_vector) == len(y_mark), "vector should be same as long as mark"
    name1 = ["y_vector"]
    df1 = pd.DataFrame({'vector': y_vector})
    df1.to_csv('data/vector.csv', encoding='utf-8')
    name2 = ["y_mark"]
    df2 = pd.DataFrame({'mark': y_mark})
    df2.to_csv('data/mark.csv', encoding='utf-8')
    end1 = time.time()
    print(end1 - start1, 's')
    start2 = time.time()
    num_ite = 0
    while True:
        y_train, y_test, y_train_mark, y_test_mark = train_test_split(y_vector, y_mark, test_size=0.2)
        clf = svm.SVC(C=0.7, kernel='rbf', degree=3, coef0=0.0,
                      probability=False, tol=0.001, cache_size=1000, class_weight='balanced', verbose=False,
                      max_iter=-1, random_state=None)
        break
        # clf.fit(y_train, y_train_mark)  # 训练模型
        # joblib.dump(clf, "data/train.raw.data.202203.pkl")  # 保存模型
        # result = clf.predict(y_test)
        # labels = [0, 1]
        # target_names = ["positive", "negative"]
        # positive_rate = \
        # classification_report(y_test_mark, result, labels=labels, target_names=target_names).split("\n")[2].split()[1]
        # negative_rate = \
        # classification_report(y_test_mark, result, labels=labels, target_names=target_names).split("\n")[3].split()[1]
        # classification_report(y_test_mark, result, labels=labels, target_names=target_names, digits=6)  # 评估分类效果
        #
        # if float(negative_rate) >= 0.8 and float(positive_rate) >= 0.8:
        #     break
        # num_ite = num_ite + 1
        # if num_ite > 100:
        #     break
        # end2 = time.time()
        # print(end2 - start2, 's')
