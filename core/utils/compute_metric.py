#-*- coding:utf-8 –*-
import numpy as np

# 计算混淆矩阵
def compute_confusion_matrix(precited,expected):
    predicted = np.array(precited,dtype = int)
    expected = np.array(expected,dtype = int)
    part = precited**expected          # 对结果进行分类，亦或使得判断正确的为0,判断错误的为1
    pcount = np.bincount(part)         # 分类结果统计，pcount[0]为0的个数，pcount[1]为1的个数
    tp_list = list(precited & expected)    # 将TP的计算结果转换为list
    fp_list = list(precited & ~expected)   # 将FP的计算结果转换为list
    tp = tp_list.count(1)                  # 统计TP的个数
    fp = fp_list.count(1)                  # 统计FP的个数
    tn = pcount[0] - tp                    # 统计TN的个数
    fn = pcount[1] - fp                    # 统计FN的个数
    return tp, fp, tn, fn

# 计算常用指标
def compute_indexes(tp, fp, tn, fn):
    accuracy = (tp+tn) / (tp+tn+fp+fn)     # 准确率
    precision = tp / (tp+fp)               # 精确率
    recall = tp / (tp+fn)                  # 召回率
    F1 = (2*precision*recall) / (precision+recall)    # F1
    return accuracy, precision, recall, F1
