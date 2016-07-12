
import utils
import sys
import datasets
import fe_extraction
from pylearn2.config import yaml_parse
import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold
import logging as log
from pylearn2.utils.rng import make_np_rng
from matplotlib import pyplot


def exploreLinearSVM(X_train, y_train, X_valid, y_valid, C_range=10.0 ** np.arange(-8, 4), scaled=False):

    #svc = svm.SVC(kernel='linear', probability=True)
    svc = svm.LinearSVC(dual=X_train.shape[0] > X_train.shape[1])
    train_scores = []
    valid_scores = []
    best_score = float('-inf')
    for C in C_range:
        svc.set_params(C=C)
        log.debug('Exploring C=%0.2e' % (C))
        svc.fit(X_train, y_train)
        train_scores.append(
            roc_auc_score(y_train, svc.decision_function(X_train)))
        valid_scores.append(
            roc_auc_score(y_valid, svc.decision_function(X_valid)))
        if valid_scores[-1] > best_score:
            best_c = C
            best_score = valid_scores[-1]
    svc.set_params(C=best_c)
    svc.fit(np.vstack((X_train, X_valid)), np.hstack((y_train, y_valid)))
    return (svc, train_scores, valid_scores)


def eval_features(X_train, y_train, X_valid, y_valid, X_test, y_test, scaled=False):
    conf = utils.get_config()
    if scaled:
        log.info('scaling features...')
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X_valid = min_max_scaler.transform(X_valid)
        X_test = min_max_scaler.transform(X_test)
    log.info('exploring svm params...')
    svc, train_scores, valid_scores = exploreLinearSVM(
        X_train, y_train, X_valid, y_valid)
    auc_train = roc_auc_score(y_train, svc.decision_function(X_train))
    auc_valid = roc_auc_score(y_valid, svc.decision_function(X_valid))
    auc_test = roc_auc_score(y_test, svc.decision_function(X_test))
    print '%i\t%s\t%i\t%f\t%f\t%f\t%0.2e' % (conf['patch_size'], conf['pool_fn'], conf['sae']['nhid'], auc_train, auc_valid, auc_test, svc.C)
    print 'AUC Train %f' % (auc_train)
    print 'AUC Valid %f' % (auc_valid)
    print 'AUC Test %f' % (auc_test)
    return svc


def boostrap(X, y, n_runs=100, C_range=10.0 ** np.arange(-8, 4), scaled=False, rng_seed=(2014, 11, 26)):
    rng = make_np_rng(None, rng_seed, which_method='uniform')
    bs = Bootstrap(len(y), n_runs, 0.8, random_state=rng)
    train_scores = np.zeros((n_runs, len(C_range)))
    valid_scores = np.zeros((n_runs, len(C_range)))
    for i, (train, valid) in enumerate(bs):
        X_t, y_t, X_v, y_v = X[train], y[train], X[valid], y[valid]
        if scaled:
            min_max_scaler = preprocessing.MinMaxScaler()
            X_t = min_max_scaler.fit_transform(X_t)
            X_v = min_max_scaler.transform(X_v)
        for j, C in enumerate(C_range):
            svc = svm.LinearSVC(dual=X_t.shape[0] > X_t.shape[1], C=C)
            svc.fit(X_t, y_t)
            train_scores[i, j] = roc_auc_score(y_t, svc.decision_function(X_t))
            valid_scores[i, j] = roc_auc_score(y_v, svc.decision_function(X_v))

    return train_scores, valid_scores


def plot_learning_curve(train_scores, valid_scores, C_range):
    pyplot.figure()
    pyplot.plot(
        np.log10(C_range), train_scores.mean(axis=0), 'b-', label='Train')
    pyplot.plot(
        np.log10(C_range), valid_scores.mean(axis=0), 'g-', label='Valid')
    pyplot.plot(np.log10(C_range), train_scores.mean(
        axis=0) - train_scores.std(axis=0), 'b--')
    pyplot.plot(np.log10(C_range), train_scores.mean(
        axis=0) + train_scores.std(axis=0), 'b--')
    pyplot.plot(np.log10(C_range), valid_scores.mean(
        axis=0) - valid_scores.std(axis=0), 'g--')
    pyplot.plot(np.log10(C_range), valid_scores.mean(
        axis=0) + valid_scores.std(axis=0), 'g--')
    pyplot.gca().set_ylim(0.4, 1)
    pyplot.legend()
    pyplot.xlabel('log10(C)')
    pyplot.ylabel('ROC_AUC')


if __name__ == '__main__':
    conf_file = sys.argv[1] if len(sys.argv) > 1 else None
    conf = utils.get_config(conf_file)
    paths = utils.get_paths()
    # eval_hand_crafted_features()
    X_train, y_train, X_valid, y_valid, X_test, y_test = fe_extraction.get_feats_from_cnn()
    eval_features(X_train, y_train, X_valid, y_valid, X_test, y_test, True)
    X_train, y_train, X_valid, y_valid, X_test, y_test = fe_extraction.get_feats_from_csv_in_partitions()
    eval_features(X_train, y_train, X_valid, y_valid, X_test, y_test, True)
    X_train, y_train, X_valid, y_valid, X_test, y_test = fe_extraction.get_feats_in_partitions()
    eval_features(X_train, y_train, X_valid, y_valid, X_test, y_test, True)
    X_train, y_train, X_valid, y_valid, X_test, y_test = fe_extraction.get_feats_from_oversampled()
    eval_features(X_train, y_train, X_valid, y_valid, X_test, y_test, True)
    X_train, y_train, X_valid, y_valid, X_test, y_test = fe_extraction.get_feats_from_imagenet_in_partitions()
    eval_features(X_train, y_train, X_valid, y_valid, X_test, y_test, True)
