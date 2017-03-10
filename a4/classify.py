"""
classify.py
"""
from collections import Counter, defaultdict
from itertools import chain, combinations
import numpy as np
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import string
from collections import defaultdict
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from io import BytesIO, StringIO
from zipfile import ZipFile
import urllib.request
import pandas as pd
import json

afinn = dict()
neg_words = []
pos_words = []
min_freqID =2

def initializeafinn():
    url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
    zipfile = ZipFile(BytesIO(url.read()))
    afinn_file = zipfile.open('AFINN/AFINN-111.txt')

    for line in afinn_file:
        parts = line.strip().split()
        if len(parts) == 2:
            afinn[parts[0].decode("utf-8")] = int(parts[1])

    for term in afinn:
        if(afinn[term]>0):
            pos_words.append(term)
        else:
            neg_words.append(term)

def afinn_sentiment(terms, afinn):
    total = 0.
    for t in terms:
        if t in afinn:
            # print('\t%s=%d' % (t, afinn[t]))
            total += afinn[t]
    return total

def tokenize(doc):
    tokens = []
    array_word = doc.split();

    for token in array_word:
        token = re.sub("^[^\w]+", '', token)
        token = re.sub("[^\w]+$", '', token)
        if token:
            tokens.append(token.lower().strip())
    trueresult = np.array(tokens)
    return trueresult

def token_features(docs_list, feats):
    counter = Counter(docs_list)
    for token in counter:
        key = "token=" + token
        feats[key] = counter[token]

def lexicon_features(tokens, feats):
    neg_words_count = 0;
    pos_words_count = 0;

    for word in tokens:
        if word.lower() in neg_words:
            neg_words_count = neg_words_count + 1
        if word.lower() in pos_words:
            pos_words_count = pos_words_count + 1

    feats['neg_words'] = neg_words_count
    feats['pos_words'] = pos_words_count

def afinn_sentiment2(terms, feats):

    neg_words_count = 0;
    pos_words_count = 0;

    for word in terms:
        if word.lower() in neg_words:
            neg_words_count = neg_words_count + 1
        if word.lower() in pos_words:
            pos_words_count = pos_words_count + 1

    feats['neg_words'] = neg_words_count
    feats['pos_words'] = pos_words_count


def featurize(tokens, feature_fns):
    feats = defaultdict(lambda: 0)
    for func in feature_fns:
        func(tokens, feats)
    return (sorted(feats.items(), key=lambda x: x[0]))


def vectorize(tokens_list, feature_fns, min_freqID, vocab=None):
    indptr = [0]
    indices = []
    data = []
    features = dict()
    masterlist = list()

    # tokens_list = [tokenize(d) for d in tweets]

    for token_list in tokens_list:
        sublist = defaultdict(lambda: 0)
        sublist = featurize(token_list, feature_fns)
        masterlist.append(sublist)


    if (vocab == None):
        for token_list in masterlist:
            for feat in token_list:
                key = feat[0]
                value = feat[1]
                if value > 0:
                    if key in features:
                        features[key] = features[key] + 1
                    else:
                        features[key] = 1

        vocabdict = {key: value for key, value in features.items() if value >= min_freqID}

        if (vocabdict):
            vocab = dict(vocabdict)
            index = 0
            for voc in sorted(vocab):
                vocab[voc] = index
                index = index + 1

            for doc in masterlist:
                for feat in sorted(doc):
                    if feat[1] > 0:
                        if feat[0] in vocab:
                            indices.append(vocab[feat[0]])
                            data.append(feat[1])
                indptr.append(len(indices))

            x = csr_matrix((data, indices, indptr), dtype='int64')
            return x, vocab
        return None, None
    else:
        for doc in masterlist:
            for feat in sorted(doc):
                if feat[1] > 0:
                    if feat[0] in vocab:
                        indices.append(vocab[feat[0]])
                        data.append(feat[1])
            indptr.append(len(indices))

        x = csr_matrix((data, indices, indptr), dtype='int64')
        return x, vocab

def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    """
    Compute the average testing accuracy over k folds of cross-validation. You
    can use sklearn's KFold class here (no random seed, and no shuffling
    needed).

    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.

    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
    ###TODO

    #referenced from Professror's notes

    cv = KFold(len(labels), k)
    accuracies = []
    for train_idx, test_idx in cv:
        #clf = LogisticRegression()
        clf.fit(X[train_idx], labels[train_idx])
        predicted = clf.predict(X[test_idx])
        acc = accuracy_score(labels[test_idx], predicted)
        accuracies.append(acc)
    avg = np.mean(accuracies)
    return avg


def eval_all_combinations(tweets, labels,
                          feature_fns, min_freqs):
    """
    Enumerate all possible classifier settings and compute the
    cross validation accuracy for each setting. We will use this
    to determine which setting has the best accuracy.

    For each setting, construct a LogisticRegression classifier
    and compute its cross-validation accuracy for that setting.

    In addition to looping over possible assignments to
    keep_internal_punct and min_freqs, we will enumerate all
    possible combinations of feature functions. So, if
    feature_fns = [token_features, token_pair_features, lexicon_features],
    then we will consider all 7 combinations of features (see Log.txt
    for more examples).

    Params:
      docs..........The list of original training documents.
      labels........The true labels for each training document (0 or 1)
      punct_vals....List of possible assignments to
                    keep_internal_punct (e.g., [True, False])
      feature_fns...List of possible feature functions to use
      min_freqs.....List of possible min_freq values to use
                    (e.g., [2,5,10])

    Returns:
      A list of dicts, one per combination. Each dict has
      four keys:
      'punct': True or False, the setting of keep_internal_punct
      'features': The list of functions used to compute features.
      'min_freq': The setting of the min_freq parameter.
      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.

      This list should be SORTED in descending order of accuracy.

      This function will take a bit longer to run (~20s for me).

    # {'features': (<function lexicon_features at 0x114172d08>,), 'punct': True, 'accuracy': 0.64749999999999996, 'min_freq': 2}
    """
    ###TODO

    k = 5
    clf = LogisticRegression()
    result = list()

    initializeafinn()

    for L in range(1, len(feature_fns) + 1):
        for subset in combinations(feature_fns, L):
            featurefns = list(subset)
            #for doc in docs:
            docs_list = [tokenize(d) for d in tweets]
            X, vocab = vectorize(docs_list, featurefns, min_freqs, vocab=None)
            if vocab:
                accuracy = cross_validation_accuracy(clf, X, labels, k)
                result.append({'features': featurefns, 'accuracy': accuracy})

    result.sort(key=lambda x:-x['accuracy'])
    return result

def getlabels(flag):
    label = []
    traindata = []

    if flag:
        url = urllib.request.urlopen('http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip')
        zipfile = ZipFile(BytesIO(url.read()))
        tweet_file = zipfile.open('testdata.manual.2009.06.14.csv')
        tweets = pd.read_csv(tweet_file,
                             header=None,
                             names=['polarity', 'id', 'date',
                            'query', 'user', 'text'])

        # label = np.array(tweets['polarity'])
        # traindata = np.array(tweets['text'])

        label = tweets['polarity']
        traindata = tweets['text']

        with open('data/labels.txt', 'w') as f:
            tweet = json.dumps(str(label))
            f.write(tweet)
            f.write('\n')
        with open('data/traindata.txt', 'w') as f:
            tweettraindata = json.dumps(str(traindata))
            f.write(tweettraindata)
            f.write('\n')
    else:
        labelfile = 'data/labels.txt'
        traindatafile='data/traindata.txt'
        with open(labelfile, 'r') as f:
            for line in f:
                label = json.loads(line)
                # label.append(item)

        with open(traindatafile, 'r') as f:
            for line in f:
                traindata = json.loads(line)
                # traindata.append(item)

        print('Labels and training data read succufully !!!')

    return label, traindata


def fit_best_classifier(traintweets, labels, best_result):
    featureID = best_result['features']
    model = LogisticRegression()
    docs_list = [tokenize(d) for d in traintweets]
    X, vocab = vectorize(docs_list, featureID, min_freqID, vocab=None)

    model.fit(X, labels)
    return model, vocab


def parse_test_data(best_result, vocab):
    # docs, labels = read_data(os.path.join('data', 'test'))
    documents = []
    traindatafile = 'data/tweets.txt'

    with open(traindatafile, 'r') as f:
        for line in f:
            docs = json.loads(line)
            documents.append(docs)

    with open('temp/classifysdata.txt', 'w') as f:
        f.write('\nNumber of messages collected\n')
        f.write(str(len(documents)))
        f.write('\n')

    featureID = best_result['features']

    docs_list = [tokenize(d) for d in documents]
    X, vocab = vectorize(docs_list, featureID, min_freqID,  vocab)
    return documents, X


def print_classified(test_docs, X_test, clf, n):
    results = list()
    example = []
    index = 0
    limit = 0
    pred_prob = clf.predict_proba(X_test)
    pred_res = clf.predict(X_test)

    while index<len(pred_res):
        tuple = {}
        tuple['pred_res'] = pred_res[index]
        tuple['text_detail'] = test_docs[index]
        tuple['probability'] = pred_prob[index]
        results.append(tuple)
        index = index + 1

    exampleclass0 = ''
    exampleclass2 = ''
    exampleclass4 = ''

    class0count = 0
    class2count = 0
    class4count = 0

    count = 0

    for result in results:
        if result['pred_res'] == 0:
            class0count = class0count +1
        elif result['pred_res'] == 2:
            class2count = class2count +1
        else:
            class4count = class4count +1

    # for result in results:
    #     if result['probability'][0] >= 0.7 and str(result['pred_res']) == '0' and exampleclass0 =='':
    #         exampleclass0 = result['text_detail']
    #         count = count + 1
    #     if result['probability'][1] >= 0.7  and str(result['pred_res']) == '2' and exampleclass2 =='':
    #         exampleclass2 = result['text_detail']
    #         count = count + 1
    #     if result['probability'][2] >= 0.7  and str(result['pred_res']) == '4' and exampleclass4 =='':
    #         exampleclass4 = result['text_detail']
    #         count = count + 1
    #     if count >= 3:
    #         break

    results = sorted(results, key=lambda x: x['probability'][0], reverse=True)
    exampleclass0 = results[0]['text_detail']

    results = sorted(results, key=lambda x: x['probability'][1], reverse=True)
    exampleclass2 = results[0]['text_detail']

    results = sorted(results, key=lambda x: x['probability'][2], reverse=True)
    exampleclass4 = results[0]['text_detail']

    output = []

    class0count = 'Class-0 Count (negative sentimates): ' + str(class0count)
    class2count = 'Class-2 count (neutral sentimates): ' + str(class2count)
    class4count = 'Class-4 count (positive sentimates): ' + str(class4count)

    output.append('\nNumber of instances per class found:')
    output.append(class0count)
    output.append(class2count)
    output.append(class4count)

    exampleclass0 = '\nClass-0 Example (negative sentimates): ' + exampleclass0
    exampleclass2 = '\nClass-2 Example (neutral sentimates): ' + exampleclass2
    exampleclass4 = '\nClass-4 Example (positive sentimates): ' + exampleclass4

    output.append('\nOne example from each class:')
    output.append(exampleclass0)
    output.append(exampleclass2)
    output.append(exampleclass4)


    with open('temp/classifysdata.txt', 'a') as f:
        for entry in output:
            f.write(entry)
            f.write('\n')


def main():
    flag = True
    clf = LogisticRegression()

    feature_fns = [token_features, afinn_sentiment2]
    labels, traintweets = getlabels(flag)

    results = eval_all_combinations(traintweets, labels,
                                    feature_fns,
                                    1)

    best_result = results[0]

    # Fit best classifier.
    clf, vocab = fit_best_classifier(traintweets, labels, results[0])

    # Parse test data
    test_docs, X_test = parse_test_data(best_result, vocab)

    print_classified(test_docs, X_test, clf, 5)

if __name__ == '__main__':
    main()