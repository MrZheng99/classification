# *-* coding:utf-8 *-*
import pandas as pd
from collections import Counter
import numpy as np
from functools import reduce


def getData(fp, class_label):
    df = pd.read_excel(fp, encoding="utf-8")
    A = [a for a in list(df.keys()) if a != class_label]
    return df, A


def getEntropy(clabelList, attrList=None):
    attr_len = len(clabelList)
    entropy = 0
    if attrList is None:
        count_dict = dict(Counter(clabelList))
        for v in count_dict.values():
            entropy += (-v / attr_len) * np.log2(v / attr_len)
    else:
        class_dict = {}
        for i in range(attr_len):
            if attrList[i] not in class_dict.keys():
                class_dict[attrList[i]] = [clabelList[i]]
            else:
                class_dict[attrList[i]].append(clabelList[i])
        for vlist in class_dict.values():
            entr = 0
            count_dict = dict(Counter(vlist))
            for v in count_dict.values():
                entr += (-v / len(vlist)) * np.log2(v / len(vlist))
            entropy += entr * len(vlist) / attr_len
    return entropy


# 离散化连续的属性
def disData(dataset, attr, c_label):
    DD = list(dataset[attr])
    D = sorted(DD)
    midValue = []
    for i in range(len(D) - 1):
        midValue.append((D[i] + D[i + 1]) / 2)
    gain = -1
    flag = midValue[0]
    for t in midValue:
        newD = D.copy()
        for i in range(len(newD)):
            newD[i] = 1 if newD[i] > t else 0
        g = getEntropy(clabelList=list(dataset[c_label]), attrList=newD)
        if gain < g:
            gain = g
            flag = t
    for i in range(len(D)):
        DD[i] = 1 if DD[i] > flag else 0
    dataset.loc[:, attr] = DD
    return dataset


def getLabelP(labelList):
    labelConP = Counter(labelList)
    P = [v / len(labelList) for v in labelConP.values()]
    return dict(zip(labelConP.keys(), P))


def getConBehind(labelConP, dfGroup, attrList, pdItem):
    AllattrMutP = {}
    for k, v in labelConP.items():
        AllattrP = []
        df = dfGroup.get_group(k)
        for attr in attrList:
            attrP = getLabelP(list(df[attr]))[pdItem[attr]]
            AllattrP.append(attrP)
        AllattrMutP[k] = v * reduce(lambda x, y: x * y, AllattrP)
    return max(AllattrMutP, key=AllattrMutP.get)


def NBC(dataset, c_label, textset):
    classList = []
    labelConP = getLabelP(list(dataset[c_label]))
    dfGroup = dataset.groupby(c_label)
    attrList = list(dataset.keys())
    del attrList[attrList.index(c_label)]
    for index, row in textset.iterrows():
        classList.append(
            getConBehind(labelConP=labelConP,
                         dfGroup=dfGroup,
                         attrList=attrList,
                         pdItem=row))
    return classList


if __name__ == '__main__':
    filename = r'data\data.xlsx'
    predictfile = r'data\data_pre.xlsx'
    c_label = "class"
    dataset, labels = getData(fp=filename, class_label=c_label)
    preset, prelabels = getData(fp=predictfile, class_label=c_label)
    max_cls = len(labels)
    for a in dataset.keys():
        if 0.3 * max_cls < len(Counter(dataset[a]).keys()):
            dataset = disData(dataset, attr=a, c_label=c_label)
            preset = disData(preset, attr=a, c_label=c_label)
    preset = preset.drop(columns=[c_label])
    preCls = NBC(dataset, c_label, preset)
    print("预测数据类别：", preCls)
