# coding> utf-8
import pickle
import matplotlib.pyplot as pyplot

from datatypes import Dataset
from classifier import naive_bayes, svm, naive_bayes_custom, knn
from feature_selection import univariate_feature_selection, lda, pca
from statistics import analize
from sklearn.cross_validation import train_test_split
from numpy import mean, std, sum, diag, shape, array

def load_spam_ds():
    """
    Loads the data from file and build the dataset in scikit format.

    () -> Dataset
    """

    data = []
    target = []
    i = 0
    with open("data/spambase.data", "r") as f:
        for line in f:
            # Removes \r\n from line
            line = line.replace("\n","").replace("\r","")
            
            items = line.split(",")
            features = [float(item) for item in items[:-1]]
            spam_class = int(items[-1])
            data.append(features)
            target.append(spam_class)
    
    return Dataset(data, target)

def split_train_test(ds):
    """
    Given the dataset, split in two datasets:
    One is the Training set. Other is the Test set.
    The proportion is 80% to 20% Respectively
    
    Dataset -> Dataset, Dataset
    """

    samples_train, samples_test, classes_train, classes_test = train_test_split(ds.data, ds.target, test_size=0.2)
    training_set = Dataset(samples_train, classes_train)
    test_set = Dataset(samples_test, classes_test)
    return training_set, test_set


def test_classifier(train_set, test_set, classifier ):
    ''' 
        Generate classifiers the normalized confusion
        matrix for a given classifier 
    '''
    classifier = classifier(train_set)
    cm = 1.0 * classifier.classify(test_set)[0] / len(test_set.data)
    return cm

def test_batch(ds, iterations = 1, classifiers = [naive_bayes_custom], configurations=[(None, 0)] ):
    results = {}
    for i in range(iterations):
        print(i, "/", iterations)
        train_set, test_set = split_train_test(ds)
        for conf_index in range(len(configurations)):
            feature_selection = configurations[conf_index][0]
            dimensions = configurations[conf_index][1]

            if feature_selection is not None :
                train, test = feature_selection(train_set, test_set, dimensions)
            else:
                train, test= train_set, test_set

            if conf_index not in results:
                results[conf_index] = {}

            for classifier in classifiers:
                if classifier not in results[conf_index]:
                    results[conf_index][classifier] = []

                confusion = test_classifier(train, test, classifier)
                results[conf_index][classifier] += [confusion]
    return results

def generate_plot(ds):
    train_set, test_set = split_train_test(ds)
    classifiers=[naive_bayes, naive_bayes_custom, knn, svm]
    feature_selection=[univariate_feature_selection, pca, lda]

    figure_index = 1
    for f in feature_selection:
        pyplot.figure(figure_index)
        figure_index += 1

        train, test = f(train_set, test_set, 2)
        subplot_index=221
        for classifier in classifiers:
            pyplot.subplot(subplot_index)
            subplot_index += 1
            pyplot.title(classifier.__name__)

            c = classifier(train)
            result = c.classify(test)[1]
            tp = (test.target == 1) & (result==1)
            tn = (test.target == 0) & (result==0)
            fp = (test.target == 0) & (result==1)
            fn = (test.target == 1) & (result==0)
            pyplot.plot(test.data[tp,0], test.data[tp,1], 'c.', alpha=0.3 )
            pyplot.plot(test.data[tn,0], test.data[tn,1], 'm.', alpha=0.5)
            pyplot.plot(test.data[fp,0], test.data[fp,1], 'bx')
            pyplot.plot(test.data[fn,0], test.data[fn,1], 'rx')
        pyplot.savefig(f.__name__+'.png')

if __name__ == "__main__":
    ds = load_spam_ds()
    generate_plot(ds)
    
    classifiers=[naive_bayes_custom, naive_bayes, knn, svm]
    configurations = [(univariate_feature_selection, 1), (pca, 1), (lda, 1), \
                    (univariate_feature_selection, 10), (pca, 10), (lda, 2), \
                    (pca,57), (None, 57)]
    configurations_labels = ["1 feature using univariate_feature_selection", \
                            "1 feature using pca", "1 feature using lda", \
                            "10 features using univariate_feature_selection", \
                            "10 features using pca", "2 features using lda", \
                            "57 features using pca", "57 features"]
    iterations = 100

    results = test_batch(ds, iterations, classifiers, configurations)
    with open('results.pickle', 'wb') as f:
        pickle.dump( (results, configurations_labels), f )

    #analize(results, configurations_labels)
    
   