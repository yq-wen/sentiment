from classifier import MultinomialNaiveBayes, SupportVectorMachine, NBSVM
import pickle
from sklearn.model_selection import cross_val_score
import pathlib
import argparse

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(
        description='Script that runs 10-fold cross validation using MNB, SVM, and NBSVM on pickled data'
    )

    arg_parser.add_argument(
        'pickled_data',
        type=str,
        help='Path to the pickled data'
    )

    args = arg_parser.parse_args()

    pickled_data = pathlib.Path(args.pickled_data).open(mode='rb')
    X, y, mapping = pickle.load(pickled_data)

    mnb = MultinomialNaiveBayes(len(mapping))
    mnb_scores = cross_val_score(mnb, X, y, cv=10)
    print('mnb: mean={}, std={}'.format(mnb_scores.mean(), mnb_scores.std()))

    svm = SupportVectorMachine(len(mapping))
    svm_scores = cross_val_score(svm, X, y, cv=10)
    print('svm: mean={}, std={}'.format(svm_scores.mean(), svm_scores.std()))

    nbsvm = NBSVM(len(mapping))
    nbsvm_scores = cross_val_score(nbsvm, X, y, cv=10)
    print('nbsvm: mean={}, std={}'.format(nbsvm_scores.mean(), nbsvm_scores.std()))
