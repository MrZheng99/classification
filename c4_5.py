import pandas as pd
from collections import Counter
import numpy as np


def getData(fp, class_label):
    df = pd.read_excel(fp, encoding="utf-8")
    A = [a for a in list(df.keys()) if a != class_label]
    return df, A


# c_label参数表示原始分类标签，attr表示是否依据某个属性对原始标签分类
# 例如在依据天气情况的好坏，对出去玩分类。
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


def getGainRatio(c_label, attrList):
    splitE = getEntropy(clabelList=attrList)
    gain = getEntropy(c_label) - getEntropy(clabelList=list(c_label),
                                            attrList=attrList)
    if splitE == 0:
        # 此处应该用小样本修正
        return gain / (splitE + 0.001)
    return gain / splitE


def majorityCnt(classList):
    '''
    数据集已经处理了所有属性，但是类标签依然不是唯一的，
    此时我们需要决定如何定义该叶子节点，在这种情况下，我们通常会采用多数表决的方法决定该叶子节点的分类
    '''
    classCont = dict(Counter(classList))
    k = sorted(classCont.items(), key=lambda x: x[1], reverse=True)[0][0]
    return k


def getBestSplitFea(dataset, labels, c_label):
    gainRatio = -1
    attr = labels[0]
    for a in labels:
        gR = getGainRatio(dataset[c_label], list(dataset[a]))
        if gainRatio < gR:
            gainRatio = gR
            attr = a
    return attr


def splitData(dataset, bestFeaLabel, value):
    dfGroup = dataset.groupby(bestFeaLabel)
    df = dfGroup.get_group(value)
    return df


def C45_createTree(dataset, labels, c_label, max_deep, lenAttrList, deep=0):
    classList = list(dataset[c_label])
    if len(classList) != lenAttrList:
        deep += 1

    if classList.count(classList[0]) == len(classList):
        # 类别完全相同，停止划分
        return classList[0]
    if dataset.shape[0] == 1 or len(labels) == 0 or deep == max_deep:
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    bestFeatLabel = getBestSplitFea(dataset, labels, c_label)
    Tree = {bestFeatLabel: {}}
    del (labels[labels.index(bestFeatLabel)])
    # 得到列表包括节点所有的属性值
    featValues = list(dataset[bestFeatLabel])
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        Tree[bestFeatLabel][value] = C45_createTree(
            splitData(dataset, bestFeatLabel, value), subLabels, c_label,
            max_deep, lenAttrList, deep)
    return Tree


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


def classfiy(tree, dataset):
    textCls = []
    for index, row in dataset.iterrows():
        firstFea = list(tree.keys())[0]
        values = tree[firstFea]
        while isinstance(values, dict):
            for k, v in values.items():
                if row[firstFea] == k:
                    values = v
                    break
            if isinstance(values, dict):
                firstFea = list(values.keys())[0]
                values = values[firstFea]
        textCls.append(values)
    return textCls


def getScore(realCls, textCls):
    TP = 0
    for i in range(len(textCls)):
        if textCls[i] == realCls[i]:
            TP += 1
    return TP / len(textCls)


if __name__ == '__main__':
    filename = r'data\0205data.xlsx'
    testfile = r'data\data_tra.xlsx'
    predictfile = r'data\data_pre.xlsx'
    c_label = "class"
    dataset, labels = getData(fp=filename, class_label=c_label)
    textset, textlabels = getData(fp=testfile, class_label=c_label)
    preset, prelabels = getData(fp=predictfile, class_label=c_label)
    max_cls = len(labels)
    for a in dataset.keys():
        if 0.3 * max_cls < len(Counter(dataset[a]).keys()):
            dataset = disData(dataset, attr=a, c_label=c_label)
            textset = disData(textset, attr=a, c_label=c_label)
            preset = disData(preset, attr=a, c_label=c_label)
    max_deep = 3
    tree = C45_createTree(dataset,
                          labels,
                          c_label,
                          max_deep,
                          lenAttrList=len(labels))
    print('desicionTree:\n', tree)
    preDataSetCls = classfiy(tree, dataset)
    realDataSetCls = list(dataset[c_label])
    traScore = getScore(realDataSetCls, preDataSetCls)
    print("训练集真实类别", realDataSetCls, "\n训练集预测类别", preDataSetCls)
    print("训练集准确率：", traScore)
    preTextSetCls = classfiy(tree, textset)
    realTextSetCls = list(textset[c_label])
    print("测试集真实类别", realTextSetCls, "\n测试集预测类别", preTextSetCls)
    textScore = getScore(realTextSetCls, preTextSetCls)
    print("测试集准确率：", textScore)
    preCls = classfiy(tree, preset)
    print("预测预测集数据类别：", preCls)
