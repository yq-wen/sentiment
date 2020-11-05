import sklearn
import numpy as np
import nltk
import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin


def log_ratio(p, q, alpha):
    '''
    Arguments:
        p (numpy.array)
        q (numpy.array)
        alpha (float): smoothing
    '''
    p += alpha
    q += alpha
    return np.log((p / np.linalg.norm(p, 1)) / (q / np.linalg.norm(q, 1)))


class LinearClassifier(BaseEstimator, ClassifierMixin):

    POS_LABEL = +1
    NEG_LABEL = -1

    def __init__(self, num_features):
        self.w = np.zeros(num_features)
        self.b = 0
        self.num_features = num_features

    def predict(self, X):
        y = []
        for i in range(X.shape[0]):
            x = X[i,:]
            if x.dot(self.w) + self.b > 0:
                y.append(LinearClassifier.POS_LABEL)
            else:
                y.append(LinearClassifier.NEG_LABEL)
        return y


class MultinomialNaiveBayes(LinearClassifier):

    def __init__(self, num_features, alpha=1):
        super().__init__(num_features)
        self.alpha = alpha

    def fit(self, X, y):
        p = np.zeros(self.num_features)
        q = np.zeros(self.num_features)
        N_pos = 0
        N_neg = 0

        for i in range(X.shape[0]):
            x = X[i, :]
            label = y[i]

            if label == LinearClassifier.POS_LABEL:
                N_pos += 1
            elif label == LinearClassifier.NEG_LABEL:
                N_neg += 1

            for idx in x.indices:
                if label == LinearClassifier.POS_LABEL:
                    p[idx] += 1
                elif label == LinearClassifier.NEG_LABEL:
                    q[idx] += 1

        self.w = log_ratio(p, q, self.alpha)
        self.b = np.log(N_pos / N_neg)


class SupportVectorMachine(LinearClassifier):

    def __init__(self, num_features, C=0.1):
        super().__init__(num_features)
        self.C = C
        self.sklearn_svm = sklearn.svm.LinearSVC(C=self.C, penalty='l2')

    def fit(self, X, y):
        self.sklearn_svm.fit(X, y)
        self.w = self.sklearn_svm.coef_.reshape(-1)
        self.b = self.sklearn_svm.intercept_

class NBSVM(LinearClassifier):

    def __init__(self, num_features, C=1, beta=0.25):
        super().__init__(num_features)
        self.C = C
        self.beta = beta
        self.mnb = MultinomialNaiveBayes(num_features)
        self.svm = SupportVectorMachine(num_features, C=self.C)
        self.r = np.zeros((1, num_features))

    def fit(self, X, y):

        self.mnb.fit(X, y)
        self.r = self.mnb.w.reshape(1, self.num_features)

        X = X.multiply(self.r)
        self.svm.fit(X, y)
        w_bar = np.linalg.norm(self.svm.w, 1) / len(self.svm.w)
        self.w = (1 - self.beta) * w_bar + self.beta * self.svm.w
        self.b = self.svm.b

    def predict(self, X):
        y = []
        for i in range(X.shape[0]):
            x = X[i,:]
            x = x.multiply(self.r)
            if x.dot(self.w) + self.b > 0:
                y.append(LinearClassifier.POS_LABEL)
            else:
                y.append(LinearClassifier.NEG_LABEL)
        return y


class LexiconClassifier(LinearClassifier):

    def __init__(self, mapping):
        '''This is a hacky implementation such that it is compatible with the
        other classifiers.
        Argument:
            mapping (dict): mapping from index to word
        '''
        self.mapping = mapping
        self.pos_words = nltk.corpus.opinion_lexicon.positive()
        self.neg_words = nltk.corpus.opinion_lexicon.negative()

    def fit(self, X, y):
        pass

    def predict(self, X):
        y = []
        for i in tqdm.trange(X.shape[0]):
            opinion = 0
            x = X[i,:]
            for idx in x.indices:
                word = self.mapping[idx]
                if word in self.pos_words:
                    opinion += 1
                elif word in self.neg_words:
                    opinion -= 1
            if opinion > 0:
                y.append(LinearClassifier.POS_LABEL)
            else:
                y.append(LinearClassifier.NEG_LABEL)
        return y
