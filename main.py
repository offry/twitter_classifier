from defines import *


# functions -------------------------------------------------------
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# -----------------------------------------------------------------

if __name__ == '__main__':
    print_hi('PyCharm')
    # open the data (specifically the first file of data) and snitizing the characters
    input_file = codecs.open("/Users/regevazran/Desktop/technion/semester g/project b/data set/Kaggle/IRAhandle_tweets_1.csv", "r", encoding='utf-8', errors='replace')

    output_file = open("/Users/regevazran/Desktop/technion/semester g/project b/data set/Kaggle/IRAhandle_tweets_1_clean.csv", "w")
    sanitize_characters(input_file, output_file)

    # saving the clean data set and printing the head
    data_set1_clean = pd.read_csv("/Users/regevazran/Desktop/technion/semester g/project b/data set/Kaggle/IRAhandle_tweets_1_clean.csv")
    print(data_set1_clean.head().to_string())

    # describe the data set
    print(data_set1_clean.describe().to_string())

    # prepare data
    clean_data_set1 = prepareData(data_set1_clean)
    print("end of prepare data")
    print(clean_data_set1["tokens"].shape)

    # split into train and test (by content and troll/not troll)
    X_train, X_test, y_train, y_test, X_train_counts, X_test_counts = split_data(clean_data_set1)
    # using tfidf
    X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    # print("start word to vec")
    # # word to vec
    # import gensim.downloader as api
    # word2vec = api.load('word2vec-google-news-300')
    # # word2vec_path = "word2vec-google-news-300"
    # # word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    # print("start embeddings")
    # embeddings = get_word2vec_embeddings(word2vec, clean_data_set1)
    # list_labels = clean_data_set1["troll_category"].fillna(' ')
    # list_labels = convert_str_lables_to_int(list_labels)
    # list_labels = list_labels.tolist()
    # list_labels = np.array(list_labels, dtype= float).tolist()
    # X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec = train_test_split(embeddings, list_labels,
    #                                                                                     test_size=0.2, random_state=40)


    # # FIXME what is happening here?
    #     all_words = [word for tokens in clean_data_set1["tokens"] for word in tokens]
    #     sentence_lengths = [len(tokens) for tokens in clean_data_set1["tokens"]]
    #     VOCAB = sorted(list(set(all_words)))
    #     print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
    #     print("Max sentence length is %s" % max(sentence_lengths))

# running regressions -------------------------------------------------
    # logistic regression with bag of words

    # logistic_regression_BOW(X_train_counts, y_train, X_test_counts, y_test, 0)
    #
    # # logistic regression with tfidf
    #
    # logistic_regression_tfidf(X_train_tfidf, y_train, X_test_tfidf, y_test, 0)
    #
    # logistic regression with word2vec
    # print("start word2vec regression")
    # logistic_regression_word2vec(X_train_word2vec, y_train_word2vec, X_test_word2vec, y_test_word2vec, 1)
    #
    # # xgboost with with bag of words
    # xgboost_regression_BOW(X_train_counts, y_train, X_test_counts, y_test, 0)
    #
    # # xgboost with tfidf
    # xgboost_regression_tfidf(X_train_tfidf, y_train, X_test_tfidf, y_test, 0)
    #
    # # decision tree with bag of words
    # tree_regression_BOW(X_train_counts, y_train, X_test_counts, y_test, 0)
    #
    # # decision tree with tfidf
    # tree_regression_tfidf(X_train_tfidf, y_train, X_test_tfidf, y_test, 0)

    # KNN with tfidf

    #KNN_regression_tfidf(X_train_tfidf, y_train, X_test_tfidf, y_test,1, 0)

# # svm
#
#     svd = decomposition.TruncatedSVD(n_components=120)
#     svd.fit(X_train_tfidf)
#     xtrain_svd = svd.transform(X_train_tfidf)
#     xvalid_svd = svd.transform(X_test_tfidf)
#
#     # Scale the data obtained from SVD. Renaming variable to reuse without scaling.
#     scl = preprocessing.StandardScaler()
#     scl.fit(xtrain_svd)
#     xtrain_svd_scl = scl.transform(xtrain_svd)
#     xvalid_svd_scl = scl.transform(xvalid_svd)
#     print("SVM: finished scale")
#     # fitting the svm
#     clf = SVC(C=1.0, probability=True)  # since we need probabilities
#     clf.fit(xtrain_svd_scl, y_train)
#     print("SVM: finished fit")
#     dump(clf, open('clf_svm.pkl', 'wb'))    # save fit clf
#     predictions = clf.predict_proba(xvalid_svd_scl)
#
#     # print("logloss: %0.3f " % multiclass_logloss(y_test, predictions)) # FIXME fix the error

# neural network with word2vec
    # print("start neural network")
    # X_train_word2vec= np.asarray(X_train_word2vec)
    # y_train_word2vec= np.asarray(y_train_word2vec)
    # net_model, cp_callback = create_network(300)
    # net_model = train_net(net_model, X_train_word2vec, y_train_word2vec, cp_callback,1)
    # X_test_word2vec= np.asarray(X_test_word2vec)
    # y_test_word2vec= np.asarray(y_test_word2vec)
    # evaluate_net(net_model, X_test_word2vec, y_test_word2vec)

# neural network with bag of words
    print("start neural network with bag of words")
    X_train_BOW = np.asarray(X_train_counts)
    y_train_BOW = np.asarray(y_train)
    net_model, cp_callback = create_network(2)
    net_model = train_net(net_model, X_train_BOW, y_train_BOW, cp_callback,1)
    X_test_BOW= np.asarray(X_test_counts)
    y_test_BOW= np.asarray(y_test)
    evaluate_net(net_model, X_test_BOW, y_test_BOW)

    #bert
    #bert(clean_data_set1)
