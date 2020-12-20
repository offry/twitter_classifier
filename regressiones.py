from defines import *


# get results of classification by logistic regression and bag of words
def get_metrics(y_test, y_predicted):
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None, average='weighted')
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None, average='weighted')
    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1


def logistic_regression_BOW(X_train, y_train, X_test, y_test, new_fit):
    clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                             multi_class='multinomial', n_jobs=-1, random_state=40)
    if new_fit == 1:
        clf.fit(X_train, y_train)  # fit new clf
        dump(clf, open('clf_logistic_beg_of_words.pkl', 'wb'))  # save fit clf

    # load the fit clf
    clf = load(open('clf_logistic_beg_of_words.pkl', 'rb'))
    predictions = clf.predict(X_test)

    # results of classification by logistic regression and bag of words
    accuracy, precision, recall, f1 = get_metrics(y_test, predictions)
    print("results of classification by logistic regression with bag of words:")
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))


def logistic_regression_tfidf(X_train, y_train, X_test, y_test, new_fit):
    clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                             multi_class='multinomial', n_jobs=-1, random_state=40)
    if new_fit == 1:
        clf.fit(X_train, y_train)  # fit new clf
        dump(clf, open('clf_logistic_tfidf.pkl', 'wb'))  # save fit clf

    # load the fit clf
    clf = load(open('clf_logistic_tfidf.pkl', 'rb'))
    predictions = clf.predict(X_test)

    # results of classification by logistic regression and tfidf
    accuracy, precision, recall, f1 = get_metrics(y_test, predictions)
    print("results of classification by logistic regression with tfidf:")
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

def logistic_regression_word2vec(X_train, y_train, X_test, y_test, new_fit):
    clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                             multi_class='multinomial', n_jobs=-1, random_state=40)
    if new_fit == 1:
        clf.fit(X_train, y_train)  # fit new clf
        dump(clf, open('clf_logistic_word2vec.pkl', 'wb'))  # save fit clf

    # load the fit clf
    clf = load(open('clf_logistic_word2vec.pkl', 'rb'))
    predictions = clf.predict(X_test)

    # results of classification by logistic regression and word2vec
    accuracy, precision, recall, f1 = get_metrics(y_test, predictions)
    print("results of classification by logistic regression with word2vec:")
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

def xgboost_regression_BOW(X_train, y_train, X_test, y_test, new_fit):
    clf_xgb = xgboost.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8,subsample=0.8, nthread=10, learning_rate=0.1)
    if new_fit == 1:
        clf_xgb.fit(X_train, y_train)
        dump(clf_xgb, open('clf_xgb_bag_of_words.pkl', 'wb'))  # save fit clf

    # load the fit clf
    clf_xgb = load(open('clf_xgb_bag_of_words.pkl', 'rb'))

    predictions = clf_xgb.predict(X_test)

    # results of xgboost regression with bag of words
    accuracy_xgboost, precision_xgboost, recall_xgboost, f1_xgboost = get_metrics(y_test, predictions)
    print("results of classification by xgboost regression with bag of words:")
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_xgboost, precision_xgboost,
                                                                           recall_xgboost, f1_xgboost))


def xgboost_regression_tfidf(X_train, y_train, X_test, y_test, new_fit):
    clf_xgb = xgboost.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8,subsample=0.8, nthread=10, learning_rate=0.1)
    if new_fit == 1:
        clf_xgb.fit(X_train, y_train)
        dump(clf_xgb, open('clf_xgb_tfidf.pkl', 'wb'))  # save fit clf

    # load the fit clf
    clf_xgb = load(open('clf_xgb_tfidf.pkl', 'rb'))

    predictions = clf_xgb.predict(X_test)

    # results of xgboost regression with bag of words
    accuracy_xgboost, precision_xgboost, recall_xgboost, f1_xgboost = get_metrics(y_test, predictions)
    print("results of classification by xgboost regression with tfidf:")
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_xgboost, precision_xgboost,
                                                                           recall_xgboost, f1_xgboost))


def tree_regression_BOW(X_train, y_train, X_test, y_test, new_fit):
    clf_tree = DecisionTreeClassifier()

    if new_fit == 1:
        clf_tree.fit(X_train, y_train)
        dump(clf_tree, open('clf_tree_bag_of_words.pkl', 'wb'))  # save fit clf

    # load the fit clf
    clf_tree = load(open('clf_tree_bag_of_words.pkl', 'rb'))

    y_predicted_tree = clf_tree.predict(X_test)

    # results of decision tree regression
    accuracy_tree, precision_tree, recall_tree, f1_tree = get_metrics(y_test, y_predicted_tree)
    print("results of classification by decision tree regression with bag of words:")
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_tree, precision_tree,
                                                                           recall_tree, f1_tree))


def tree_regression_tfidf(X_train, y_train, X_test, y_test, new_fit):
    clf_tree = DecisionTreeClassifier()

    if new_fit == 1:
        clf_tree.fit(X_train, y_train)
        dump(clf_tree, open('clf_tree_tfidf.pkl', 'wb'))  # save fit clf

    # load the fit clf
    clf_tree = load(open('clf_tree_tfidf.pkl', 'rb'))

    y_predicted_tree = clf_tree.predict(X_test)

    # results of decision tree regression
    accuracy_tree, precision_tree, recall_tree, f1_tree = get_metrics(y_test, y_predicted_tree)
    print("results of classification by decision tree regression with tfidf:")
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_tree, precision_tree,
                                                                           recall_tree, f1_tree))


def KNN_regression_tfidf(X_train, y_train, X_test, y_test,neighbors, new_fit):
    clf_KNN = KNeighborsClassifier(n_neighbors=neighbors)

    if new_fit == 1:
        clf_KNN.fit(X_train, y_train)
        dump(clf_KNN, open('clf_KNN_tfidf.pkl', 'wb'))  # save fit clf

    # load the fit clf
    clf_KNN = load(open('clf_KNN_tfidf.pkl', 'rb'))

    y_predicted_KNN = clf_KNN.predict(X_test)

    # results of decision KNN regression
    accuracy_KNN, precision_KNN, recall_KNN, f1_KNN = get_metrics(y_test, y_predicted_KNN)
    print("results of classification by decision KNN regression with tfidf and ",neighbors," neghbors:")
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_KNN, precision_KNN,
                                                                           recall_KNN, f1_KNN))
