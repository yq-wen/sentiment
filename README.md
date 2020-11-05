# Sentiment

This is an attempt to implement the methods (NB, SVM, and NBSVM) described in [Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf).
The classifiers are implemented in [classifer.py](src/classifier.py), and they are intended to be compatible with sklearn such that you can run k-fold cross validations with sklearn. [run.py](src/run.py) demostrates how this is done.

## How the classifiers work

Classifiers in [classifer.py](src/classifier.py) implements two methods: fit and predict.
They are general and not specific to sentiment analysis.
However, under the context of reproducing the [paper](ttps://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf), this means that when fitting, X is expected to be a sparse matrix with a shape of (num_data, num_features) that contains binary features that encodes the text. On the other hand, y is expected to be an array of shape (num_data,) that contains the labels (-1 for negative, +1 for positive) of the text.

## Running Instructions

You can set up a virtual environment and install all the dependencies by running:

```
pip3 install venv
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

Then, you can simply run the script with a preprocessed pickled data:
```
python3 classifier.py ../pickled_data/\(X\,\ y\,\ mapping\)_rts_uni
```

## Preprocessed Data
[Pickled data](src/pickled_data) are preprocessed data that are dumped with the pickle library.
They are 3-tuples (X, y, mapping) where X is the sparse matrix that encodes the dataset, y is the labels, and mapping is a dictionary that maps grams to indices in the binary feature vector.
Currently, the folder contains the preprocessed data from RT-s and RT-2k with both unigrams and bigrams.
