# -*- encoding: utf-8 -*-
import MySQLdb as mysqldb
import jieba
import scipy.sparse as sparse
import re


def query(sql):
    dbfile = 'datamining'
    con = mysqldb.connect(host='localhost', port=3306, user='root', passwd='', db=dbfile, charset='UTF8')
    cur = con.cursor()  # 获取游标

    # print(sql)
    cur.execute(sql)

    resultSet = cur.fetchall()
    cur.close()
    con.close()
    return resultSet


def toData(train_resultset, test_resultset):
    corpus_train = []  # 训练集中的每条短信分词，去重后组成的list

    target_train = []  # 训练集中的每条短信所属的分类： '1'或 '0'

    for line in train_resultset:
        array = jieba.cut(line[-1])
        # newString = ''.join(array)  #



        newSet = set()  # 为了过滤掉重复的词

        for i in array:
            newSet.add(i + ' ')  # 加空格，把不同的词隔开
            # print(i)
        # corpus_train.append(a for a in array)
        newArray = list(newSet)  # 形成字符串列表
        newString = ''.join(newArray)
        # print(newString)
        corpus_train.append(newString)
        # target_train.append(line[1] for a in array)
        target_train.append(line[1])

    # print('corpus_train : ')
    # print(corpus_train)
    # print('target_train length: ')
    # print(len(target_train))
    corpus_test = []  # 测试集中的每条短信分词，去重后组成的list

    target_test = []  # 训练集中的每条短信所属的分类： '1'或 '0'，无意义，现已弃用
    for line in test_resultset:
        array = jieba.cut(line[-1])
        # newString = ''.join(array)
        newSet = set()  # 为了过滤掉重复的词
        for i in array:
            newSet.add(i + ' ')  # 加空格，把不同的词隔开
            # print(i)

        newArray = list(newSet)  # 形成字符串列表
        newString = ''.join(newArray)  # 重新转化为字符串
        # print(newString)
        corpus_test.append(newString)

    return [[corpus_train, target_train], [corpus_test, target_test]]
