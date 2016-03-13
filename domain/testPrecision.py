# -*- encoding: utf-8 -*-
if __name__ == '__main__':
    # from testSklearn import testSklearn
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import VotingClassifier  # for sklearn 0.17
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    import time
    from toData import toData
    from toData import query
    from sklearn import svm
    from sklearn import tree
    import random
    from sklearn import metrics
    from sklearn.grid_search import GridSearchCV
    from  sklearn.cluster import KMeans
    from sklearn.pipeline import Pipeline
    from sklearn.semi_supervised import LabelSpreading

    startTime = time.time()

    # train_resultset = query('train1k')
    # test_resultset = query('predict1k')

    train_sql = 'select * from trainmessage limit 0,80000 '
    train_resultset = query(train_sql)
    begin_number = random.randint(100000, 770000)
    predict_sql = 'select * from trainmessage limit ' + str(begin_number) + ' ,20000'
    print('from:' + str(begin_number))
    test_resultset = query(predict_sql)

    [[corpus_train, target_train], [corpus_test, target_test]] = toData(train_resultset, test_resultset)
    # train
    # print(corpus_train)
    # stopWordsArray = []
    # for word in query('select stop_word from stop_words'):
    #     stopWordsArray.append(word)
    step1 = time.time()
    print('step1: ' + str(step1 - startTime))

    count_v1 = CountVectorizer()  # Convert a collection of raw documents to a matrix of TF-IDF features.
    counts_train = count_v1.fit_transform(corpus_train)  # 得到词频矩阵

    transformer = TfidfTransformer()  # Transform a count matrix to a normalized tf or tf-idf representation
    tfidf_train = transformer.fit_transform(counts_train)  # 得到频率矩阵

    # weight_train = tfidf_train.toarray()  # weight[i][j],第i个文本，第j个词的tf-idf值
    step2 = time.time()
    print('step2:' + str(step2 - step1))
    # predict

    counts_test = count_v1.transform(corpus_test)  # fit_transform是将文本转为词频矩阵

    tfidf_test = transformer.transform(counts_test)  # fit_transform是计算tf-idf

    # print("tfidf_test:" + str(tfidf_test.get_shape()))
    # weight_test = tfidf_test.toarray()  # weight[i][j],第i个文本，第j个词的tf-idf值
    step3 = time.time()
    print('step3:' + str(step3 - step2))

    # bayes模型
    bayes = MultinomialNB(alpha=0.15)
    # #
    bayes.fit(tfidf_train, target_train)
    bayes_pred = bayes.predict(tfidf_test)
    # use pipeline
    # pipe = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
    # pipe.set_params(clf__alpha=0.126)
    # pipe.fit(corpus_train, target_train)
    # pipe_pred = pipe.predict(corpus_test)

    # knn模型
    # 数据量大会崩
    # knnclf = KNeighborsClassifier()
    # knnclf.fit(tfidf_train, target_train)
    # knn_pred = knnclf.predict(tfidf_test)
    # svm模型
    # 慢，占用很多CPU
    # svc = svm.SVC(kernel='linear')
    '''
     implemented in terms of liblinear rather than libsvm，scale better to large numbers of
     samples.
     '''
    svc = svm.LinearSVC()  # 和上面的等价，但能大幅缩短运行时间
    # svc.fit(tfidf_train, target_train)
    # svc_pred = svc.predict(tfidf_test)
    # tree模型
    tre = tree.DecisionTreeClassifier()
    # tre.fit(tfidf_train, target_train)
    # tre_pred = tre.predict(tfidf_test)
    # LR
    # lr = LogisticRegression()
    # lr.fit(tfidf_train, target_train)
    # lr_pred = lr.predict(tfidf_test)
    # 随机森林
    # rf = RandomForestClassifier(random_state=1)
    # rf.fit(tfidf_train, target_train)
    # rf_pred=rf.predict(tfidf_test)

    # 组合分类器
    # Voting
    vote = VotingClassifier(estimators=[('nb', bayes), ('dt', tre), ('svm', svc)], voting='hard')
    vote.fit(tfidf_train, target_train)
    vote_pred = vote.predict(tfidf_test)

    # Adaboost
    # ab = AdaBoostClassifier()
    # ab.fit(tfidf_train, target_train)
    # ab_pred = ab.predict(tfidf_test)
    # Bagging
    # bag = BaggingClassifier()
    # bag.fit(tfidf_train, target_train)
    # bag_pred = bag.predict(tfidf_test)
    # Gradient
    # MemoryError，已弃用
    # gb = GradientBoostingClassifier()
    # gb.fit(tfidf_train, target_train)
    # gb_pred = gb.predict(tfidf_test.toarray())
    # 非监督
    # KMeans
    # kmeans = KMeans(n_clusters=2, n_init=3)  # TODO
    # kmeans.fit(tfidf_train)
    # kmeans_pred = kmeans.predict(tfidf_test)
    # print(kmeans.inertia_)
    # 半监督
    # Semi_Supervised

    # semi = LabelSpreading(kernel='knn',n_neighbors=1)
    # semi.fit(corpus_train, target_train)
    # semi_pred = semi.predict(corpus_test)
    #
    # result_classify = [item[1] for item in test_resultset]
    # print(metrics.classification_report(result_classify, bayes_pred))
    # print(metrics.confusion_matrix(result_classify, bayes_pred))

    # parameters = {'clf__alpha': [float(i)/10000 for i in range(1250,1270,1)], }
    # gs_clf = GridSearchCV(pipe, parameters, n_jobs=4)
    #
    # gs_clf = gs_clf.fit(corpus_train, target_train)
    # gs_pred = gs_clf.predict(corpus_test)
    # best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
    # for param_name in sorted(parameters.keys()):
    #     print("%s: %r" % (param_name, best_parameters[param_name]))
    # print("score: " + str(gs_clf.best_score_))

    step4 = time.time()
    print('step4:' + str(step4 - step3))
    i = 0

    predict_rubbish_message = 0
    predict_normal_message = 0
    actual_rubbish_message = 0
    actual_normal_message = 0
    a = 0
    b = 0
    c = 0
    d = 0
    for element in vote_pred:
        # print(element)
        if str(element) == '1':  # predict rubbish
            # print(test_resultset[i][-1].encode('utf-8'))
            predict_rubbish_message += 1
            if test_resultset[i][1] == '1':  # actual rubbish
                d += 1
            else:  # actual normal
                b += 1

        else:  # predict normal
            predict_normal_message += 1
            if test_resultset[i][1] == '1':  # actual rubbish
                c += 1
            else:  # actual normal
                a += 1

        if test_resultset[i][1] == '1':  # actual rubbish
            actual_rubbish_message += 1
        else:  # actual normal
            actual_normal_message += 1

        i += 1
    # print(b, d)
    F_rubbish = 0.65 * float(d) / (b + d) + 0.35 * float(d) / (c + d)
    F_normal = 0.65 * float(a) / (a + c) + 0.35 * float(a) / (a + b)
    F_total = 0.7 * F_rubbish + 0.3 * F_normal
    print('total message: ' + str(i))
    # print("predict rubbish: " + str(predict_rubbish_message))
    # print('predict normal: ' + str(predict_normal_message))
    # print("actual rubbish: " + str(actual_rubbish_message))
    # print('actual normal:  ' + str(actual_normal_message))
    # print('predict normal and actual normal: ' + str(a))
    # print('predict rubbish and actual normal: ' + str(b))
    # print('predict normal and actual rubbish: ' + str(c))
    # print('predict rubbish and actual rubbish: ' + str(d))
    # print('F rubbish: ' + str(F_rubbish))
    # print('F normal: ' + str(F_normal))
    print('F total: ' + str(F_total))
    endTime = time.time()
    step5 = time.time()
    print('step5:' + str(step5 - step4))
    print('use time: ' + str(endTime - startTime))
