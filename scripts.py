"""Helper scripts for 'Predicting Pop Subgenres'"""


import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

def print_metrics(labels, preds, title=None):
    '''This function prints evaluation metric scores of two series (labels, predictions)'''
    print(f"{title} Accuracy Score: {round(accuracy_score(labels, preds),4)}")
    print(f"{title} Precision Score: {round(precision_score(labels, preds),4)}")
    print(f"{title} Recall Score: {round(recall_score(labels, preds),4)}")
    print(f"{title} F1 Score: {round(f1_score(labels, preds),4)}")


def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.Greens):
    """Add Normalization Option."""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)
    fmt = '.0%' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt, ),
                 horizontalalignment="center", fontsize=18, fontweight='bold',
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Actual Tag', fontsize=15)
    plt.xlabel('Predicted Tag', fontsize=15)
    plt.grid(None)
    

def create_plot_of_feature_importances(model, X):
    ''' 
    Inputs: 

    model: A trained ensemble model instance
    X: a dataframe of the features used to train the model
    '''

    feat_importances = model.feature_importances_

    features_and_importances = zip(X.columns, feat_importances)
    features_and_importances = sorted(features_and_importances,
                                      key=lambda x: x[1], reverse=True)

    features = [i[0] for i in features_and_importances]
    importances = [i[1] for i in features_and_importances]

    plt.figure(figsize=(10, 6))
    plt.barh(features[:20], importances[:20])
    plt.gca().invert_yaxis()
    plt.title('Top 20 determining features')
    plt.xlabel('importance')