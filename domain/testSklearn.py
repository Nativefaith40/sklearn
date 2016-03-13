# -*- encoding: utf-8 -*-
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
# 用文本文档来构建数值特征向量
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier  # for sklearn 0.17
import time
from toData import toData
from toData import query
from sklearn import svm
from sklearn import tree
startTime = time.time()

# train_resultset = query('train1k')
# test_resultset = query('predict1k')
train_sql = 'select * from trainmessage'
predict_sql = 'select * from predictmessage'
train_resultset = query(train_sql)  # 获取训练集
test_resultset = query(predict_sql)  # 获取测试集

# 数据预处理
[[corpus_train, target_train], [corpus_test, target_test]] 
  = toData(train_resultset, test_resultset)

# stopWordsArray = []
# for word in query('select stop_word from stop_words'):
#     stopWordsArray.append(word)
step1 = time.time()
print('step1: ' + str(step1 - startTime))
# 针对训练集  Tokenizing
# 加停用词，12.7，准确率无变化
# count_v1 = CountVectorizer(
#     stop_words=stopWordsArray)  # Convert a collection of raw documents to a matrix of TF-IDF features.
count_v1 = CountVectorizer()                         # Convert a collection of raw documents to a matrix of TF-IDF features.
counts_train = count_v1.fit_transform(corpus_train)  # 得到词频矩阵
# for s in count_v1.get_feature_names():
#     print(s)
transformer = TfidfTransformer()                      # Transform a count matrix to a normalized tf or tf-idf representation
tfidf_train = transformer.fit_transform(counts_train)  # 得到频率矩阵
# weight_train = tfidf_train.toarray()  # weight[i][j],第i个文本，第j个词的tf-idf值

step2 = time.time()
print('step2:' + str(step2 - step1))
# 针对测试集    Tokenizing

# 不需要重复构造分类器，已弃用
# count_v2 = CountVectorizer(vocabulary=count_v1.vocabulary_)  # 让两个CountVectorizer共享vocabulary

# counts_test = count_v2.fit_transform(corpus_test)  # fit_transform是将文本转为词频矩阵  不需要重复训练，已弃用
counts_test = count_v1.transform(corpus_test)                                # fit_transform是将文本转为词频矩阵  # transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
tfidf_test = transformer.transform(counts_test)  # fit_transform是计算tf-idf
# weight_test = tfidf_test.toarray()  # weight[i][j],第i个文本，第j个词的tf-idf值

step3 = time.time()
print('step3:' + str(step3 - step2))
# bayes模型
bayes = MultinomialNB(alpha=0.15)               
# svm模型
svc = svm.LinearSVC()
# tree模型
tre = tree.DecisionTreeClassifier()
# 组合分类器
# Voting
vote = VotingClassifier(estimators=[('nb', bayes), ('svm', svc), ('tre', tre)], voting='hard')
vote.fit(tfidf_train, target_train)
vote_pred = vote.predict(tfidf_test)

step4 = time.time()
print('step4:' + str(step4 - step3))

i = 0
resultFile = open('result2.csv', 'w')  # 结果文件
for element in vote_pred:  # 遍历预测结果
    resultFile.write(str(test_resultset[i][0]) + ',' + str(element) + '\n')
    i += 1
print(counter)
resultFile.close()  # 关闭结果文件

# fileTrain.close()
step5 = time.time()
print('step5:' + str(step5 - step4))
endTime = time.time()
print('use time: ' + str(endTime - startTime))  # testSklearn()
