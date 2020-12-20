from defines import *


def sanitize_characters(input_file, output_file):
    for line in input_file:
        out = line
        output_file.write(line)


# replacing some characters and removing capital letters
def standardize_text(df, content_field):
    df[content_field] = df[content_field].str.replace(r"http\S+", "")
    df[content_field] = df[content_field].str.replace(r"http", "")
    df[content_field] = df[content_field].str.replace(r"@\S+", "")
    df[content_field] = df[content_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[content_field] = df[content_field].str.replace(r"@", "at")
    df[content_field] = df[content_field].str.lower()
    return df


# merge different categories to troll and not-troll
def merge_categories(df, content_field):
    df[content_field] = df[content_field].str.replace(r"Commercial", "Not Troll")
    df[content_field] = df[content_field].str.replace(r"Fearmonger", "Not Troll")
    df[content_field] = df[content_field].str.replace(r"HashtagGamer", "Not Troll")
    df[content_field] = df[content_field].str.replace(r"NewsFeed", "Not Troll")
    df[content_field] = df[content_field].str.replace(r"NonEnglish", "Not Troll")
    df[content_field] = df[content_field].str.replace(r"Unknown", "Not Troll")
    df[content_field] = df[content_field].str.replace(r"RightTroll", "Troll")
    df[content_field] = df[content_field].str.replace(r"LeftTroll", "Troll")
    return df

def convert_str_lables_to_int(df):
    df = df.str.replace(r"Not Troll", "0")
    df = df.str.replace(r"Troll", "1")

    return df

def cv(data):
    count_vectorized = CountVectorizer()

    emb = count_vectorized.fit_transform(data)

    return emb, count_vectorized


def prepareData(data):
    # replacing some characters and removing capital letters
    data = standardize_text(data, "content")

    data.to_csv("clean_data.csv")
    # adding another category of troll and not troll which with we are going to work
    clean_data_set = pd.read_csv("clean_data.csv")
    troll_category = [None] * len(clean_data_set.account_category);
    clean_data_set['troll_category'] = clean_data_set['account_category'];
    clean_data_set = merge_categories(clean_data_set, "troll_category")

    # adding another column of cutting the tweets in to separate tokens of words
    tokenizer = RegexpTokenizer(r'\w+')
    clean_data_set["tokens"] = clean_data_set["content"].apply(str).apply(tokenizer.tokenize)

    return clean_data_set


def split_data(data):
    list_corpus = data["content"].fillna(' ').tolist()
    list_labels = data["troll_category"].fillna(' ').tolist()
    X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2,
                                                        random_state=40)
    X_train_counts, count_vectorized = cv(X_train)
    X_test_counts = count_vectorized.transform(X_test)



    return X_train, X_test, y_train, y_test, X_train_counts, X_test_counts


# using tfidf
def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()

    train = tfidf_vectorizer.fit_transform(data)

    return train, tfidf_vectorizer


def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):

    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_questions, generate_missing=False):
    embeddings = clean_questions['tokens'].apply(lambda x: get_average_word2vec(x, vectors,
                                                                                generate_missing=generate_missing))
    return list(embeddings)