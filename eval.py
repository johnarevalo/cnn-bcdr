from pylearn2.utils import serial
import utils
import classify
import fe_extraction
import numpy as np
from pylearn2.utils.rng import make_np_rng
from sklearn import preprocessing
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from matplotlib import pyplot
import sys


def get_features(rows, features, segm_ids):
    '''
    Get features from a set of rows using segm_ids as a LUT.
    '''
    X = np.zeros((len(rows), features.shape[1]))
    y = np.zeros(len(rows))
    for i, row in enumerate(rows):
        X[i] = features[np.nonzero(int(row['segmentation_id']) == segm_ids)][0]
        y[i] = utils.is_positive(row)
    return X, y


rng = [2014, 12, 5]
rng = make_np_rng(None, rng, which_method='uniform')
scale_feats = True
n_runs = 20
C_range = 10.0 ** np.arange(-8, 8)
train_scores = np.zeros((n_runs, len(C_range)))
valid_scores = np.zeros((n_runs, len(C_range)))
fit_threshold = True
conf_file = sys.argv[1] if len(sys.argv) > 1 else None
conf = utils.get_config(conf_file)
features = np.empty([len(utils.load_csv()), 0])
#f_list = ['hcfeats', 'imnet', 'cnn']
f_list = ['cnn']

if 'imnet' in f_list:
    rows = utils.load_csv()
    feats, y = fe_extraction.get_feats_from_imagenet(rows)
    features = np.hstack((features, feats))
    segm_ids = np.asarray([int(row['segmentation_id']) for row in rows])
if 'hcfeats' in f_list:
    rows = utils.load_csv(conf['csv_features_file'])
    feats, y = fe_extraction.get_feats_from_csv(
        rows, prefixes=['s_', 't_', 'i_'])
    feats = np.asarray(feats)
    features = np.hstack((features, feats))
    segm_ids = np.asarray([int(row['segmentation_id']) for row in rows])
if 'cnn' in f_list:
    cnn_layer = 'cnn_layer_%i' % (conf['cnn_layers'])
    paths = utils.get_paths(conf)
    model_path = paths[cnn_layer]['best_model']
    model = serial.load(model_path)
    rows = utils.load_csv()
    chunkSize = 32
    feats, y = (None, None)
    for i in range(0, len(rows), chunkSize):
        offset = min(i + chunkSize, len(rows))
        f_chunk, y_chunk = fe_extraction.get_feats_from_cnn(
            rows[i:offset], model)
        if feats is None:
            feats = f_chunk
            y = y_chunk
        else:
            feats = np.vstack((feats, f_chunk))
            y = np.hstack((y, y_chunk))

    segm_ids = np.asarray([int(row['segmentation_id']) for row in rows])
    features = np.hstack((features, feats))

train_rows, valid_rows, test_rows = utils.split_dataset(
    utils.get_filtered_rows(), conf['valid_percent'],
    conf['test_percent'], rng=conf['rng_seed'])

rows = train_rows + valid_rows
patients = utils.rows_to_patients(rows)
for i in range(n_runs):
    train_rows, empty_rows, valid_rows = utils.split_dataset(
        rows, valid_percent=0, test_percent=0.2, rng=rng, patients=patients)
    X_train, y_train = get_features(train_rows, features, segm_ids)
    X_valid, y_valid = get_features(valid_rows, features, segm_ids)
    print 'train: %i, valid: %i' % (X_train.shape[0], X_valid.shape[0])

    if scale_feats:
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X_valid = min_max_scaler.transform(X_valid)
    for j, C in enumerate(C_range):
        print 'explore %i-th run with %i-th param...' % (i, j)
        svc = svm.LinearSVC(dual=X_train.shape[0] > X_train.shape[1], C=C)
        #svc = LogisticRegression(dual=X_train.shape[0] > X_train.shape[1], C=C, tol=1e-6)
        svc.fit(X_train, y_train)
        train_scores[i, j] = metrics.roc_auc_score(
            y_train, svc.decision_function(X_train))
        valid_scores[i, j] = metrics.roc_auc_score(
            y_valid, svc.decision_function(X_valid))

classify.plot_learning_curve(train_scores, valid_scores, C_range)
pyplot.savefig('learning_curve.png')
#best = valid_scores.std(axis=0).argmin()
best = valid_scores.mean(axis=0).argmax()
print '%i\t%f\t%0.3e\t%f\t%0.3e' % (features.shape[1],
                                    train_scores.mean(axis=0)[best],
                                    train_scores.std(axis=0)[best],
                                    valid_scores.mean(axis=0)[best],
                                    valid_scores.std(axis=0)[best])

X_train, y_train = get_features(rows, features, segm_ids)
X_test, y_test = get_features(test_rows, features, segm_ids)
if scale_feats:
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)

svc = svm.SVC(C=C_range[best], kernel='linear')
svc.fit(X_train, y_train)
score = svc.decision_function(X_train)
auc = metrics.roc_auc_score(y_train, score)

if fit_threshold:
    # Choose the threshold that maximizes f1
    pr, rc, thresholds = metrics.precision_recall_curve(y_train, score)
    thr = thresholds[np.argmax(2 * pr * rc / (pr + rc))]
    y_pred = score > thr
else:
    y_pred = svc.predict(X_train)

tn, fp, fn, tp = metrics.confusion_matrix(y_train, y_pred).flatten()
train_f1 = metrics.f1_score(y_train, y_pred)
print '%0.3e\t%f\t%f\t%f\t%f\t%f' % (C_range[best], tp, tn, fp, fn, auc)

score = svc.decision_function(X_test)
if fit_threshold:
    y_pred = score > thr
else:
    y_pred = svc.predict(X_test)
auc = metrics.roc_auc_score(y_test, score)

tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).flatten()
test_f1 = metrics.f1_score(y_test, y_pred)
print '%0.3e\t%f\t%f\t%f\t%f\t%f' % (C_range[best], tp, tn, fp, fn, auc)
