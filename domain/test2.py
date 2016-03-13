import jieba
import os
# import sys
import codecs
from sklearn import feature_extraction
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB


# --------------#
def load_data():
    corpus_train = []  # 字典
    target_train = []  # 分类
    filepath = 'E:python_pananteng/程序6：文本挖掘/文本分类/实例2/train'
    filelist = os.listdir(filepath)
    for num in range(len(filelist)):
        filetext = filepath + "/" + filelist[num]
        filename = os.path.basename(filetext)
        myfile = codecs.open(filetext, 'r', 'utf-8')
        temp = myfile.readlines()  # 文本内容
        myfile.close()
        for i in range(0, 100):
            len_0 = len(temp)
            seg_list = jieba.cut(','.join(temp[int(i * len_0 / 100):int((i + 1) * len_0 / 100)]), cut_all=False)
            words = " ".join(seg_list)  # ??????
            target_train.append(filename)
            corpus_train.append(words)
            # --------------#
            corpus_test = []
            target_test = []
            filepath = 'E:python_pananteng/程序6：文本挖掘/文本分类/实例2/test'
            filelist = os.listdir(filepath)
        for num in range(len(filelist)):
            filetext = filepath + "/" + filelist[num]
            myfile = open(filetext, 'r')
            temp = myfile.readlines()
            myfile.close()
            seg_list = jieba.cut(','.join(temp[1:]), cut_all=False)
            target_test.append(temp[0])
            corpus_test.append(words)
    return [[corpus_train, target_train], [corpus_test, target_test]]


# --------------#
def data_pro():
    [[corpus_train, target_train], [corpus_test, target_test]] = load_data()
    count_v1 = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    counts_train = count_v1.fit_transform(corpus_train)  # fit_transform是将文本转为词频矩阵
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf_train = transformer.fit(counts_train).transform(counts_train)  # fit_transform是计算tf-idf
    weight_train = tfidf_train.toarray()  # weight[i][j],第i个文本，第j个词的tf-idf值
    count_v2 = CountVectorizer(vocabulary=count_v1.vocabulary_)  # 让两个CountVectorizer共享vocabulary
    counts_test = count_v2.fit_transform(corpus_test)  # fit_transform是将文本转为词频矩阵
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf_test = transformer.fit(counts_train).transform(counts_test)  # fit_transform是计算tf-idf
    weight_test = tfidf_test.toarray()  # weight[i][j],第i个文本，第j个词的tf-idf值
    return [[weight_train, target_train], [weight_test, target_test]]
    # --------------#


[[weight_train, target_train], [weight_test, target_test]] = data_pro()
# ---------------------------------------------#
# knn模型
knnclf = KNeighborsClassifier()
knnclf.fit(weight_train, target_train)
knn_pred = knnclf.predict(weight_test)
# ---------------------------------------------#
# ---------------------------------------------#
# svm模型
svc = svm.SVC(kernel='linear')
svc.fit(weight_train, target_train)
svc_pred = svc.predict(weight_test)
# ---------------------------------------------#
# ---------------------------------------------#
# tree模型
tre = tree.DecisionTreeClassifier()
tre.fit(weight_train, target_train)
tre_pred = tre.predict(weight_test)
# ---------------------------------------------#
# ---------------------------------------------#
# bayes模型
bayes = MultinomialNB(alpha=0.01)
bayes.fit(weight_train, target_train)
bayes_pred = bayes.predict(weight_test)
# ---------------------------------------------#
