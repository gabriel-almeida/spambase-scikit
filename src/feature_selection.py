# coding: utf-8

from datatypes import Dataset

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.decomposition import PCA

def univariate_feature_selection(train_set, test_set, n):
    """
    Selects 'n' features in the dataset. Returns the Reduced Dataset
    n (int), ds (Dataset) -> Dataset
    """
    selector = SelectKBest(f_classif, n)
    selector.fit(train_set.data, train_set.target)
    train = Dataset(selector.transform(train_set.data), train_set.target)
    test = Dataset(selector.transform(test_set.data), test_set.target)
    return train, test

def lda(train_set, test_set, n):
    '''
        Outputs the projection of the data in the best
        discriminant dimension.
        Maximum of 2 dimensions for our binary case (values of n greater than this will be ignored by sklearn)
    '''
    selector = LDA(n_components=n)
    selector.fit(train_set.data, train_set.target)
    train = Dataset(selector.transform(train_set.data), train_set.target)
    test = Dataset(selector.transform(test_set.data), test_set.target)
    return train, test

def pca(train_set, test_set, n):
    '''
        Uses the PCA classifier to reduces the dimensionality by choosing the n lastest elements
        of the transform.
    '''
    selector = PCA(n_components=n)
    selector.fit(train_set.data, train_set.target)
    train = Dataset(selector.transform(train_set.data), train_set.target)
    test = Dataset(selector.transform(test_set.data), test_set.target)
    return train, test