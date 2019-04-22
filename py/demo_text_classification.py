# -*- coding:utf-8 -*-
# Author：hankcs
# Date: 2018-05-23 17:26
import os

from pyhanlp import SafeJClass
from tests.test_utility import ensure_data

# 实例化朴素贝叶斯分类器实例
NaiveBayesClassifier = SafeJClass('com.hankcs.hanlp.classification.classifiers.NaiveBayesClassifier')
# 实例化文件IO实例
IOUtil = SafeJClass('com.hankcs.hanlp.corpus.io.IOUtil')
# 自己的微博训练语料(这里如果Python是用HanLP安装的，请将训练文件夹放在下边的路径下
# C:\ProgramData\Anaconda3\Lib\site-packages\pyhanlp\static\data\test
sentiment_corpus_path = ensure_data('新浪微博/train', '')

def train_or_load_classifier():
    # 朴素贝叶斯模型文件名
    model_path = sentiment_corpus_path + '.ser'
    # 检查模型文件是否存在，如果存在则加载模型并返回朴素贝叶斯分类器对象
    if os.path.isfile(model_path):
        return NaiveBayesClassifier(IOUtil.readObjectFrom(model_path))
    # 模型文件不存在，则首先构建朴素贝叶斯分类器实例
    classifier = NaiveBayesClassifier()
    # 传入训练文件路径名称进行训练
    classifier.train(sentiment_corpus_path)
    # 获取训练后得到的模型
    model = classifier.getModel()
    # 保存模型为模型文件
    IOUtil.saveObjectTo(model, model_path)
    # 传入模型到朴素贝叶斯分类器并返回朴素贝叶斯分类器对象
    return NaiveBayesClassifier(model)


def predict(classifier, text):
    print("《%16s》\t属于分类\t【%s】" % (text, classifier.classify(text)))
    # 如需获取离散型随机变量的分布，请使用predict接口
    # print("《%16s》\t属于分类\t【%s】" % (text, classifier.predict(text)))


if __name__ == '__main__':
    run_mode = 'batch'
    # 训练或加载朴素贝叶斯分类器
    classifier = train_or_load_classifier()
    if run_mode == 'debug':
        while True:
            text = input("请输入文本:\n")
            if text == "q":
                break
            predict(classifier, text)
    else:
        # 预测
        predict(classifier, "运筹帷幄，高瞻远瞩，大国设计师！主席您辛苦了，向您致敬！[赞][赞][赞]")
        predict(classifier, "一张都没有看对。很郁闷。是不是手机玩多了。")
        predict(classifier, "麻城龟山人路过，给家乡大大的赞[爱慕][玫瑰]")
