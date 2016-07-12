import utils
from pylearn2.utils import serial
import theano.tensor as T
import theano
from theano.tensor.nnet import conv
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from sklearn.metrics import roc_auc_score
import numpy as np
import logging as log
from sklearn import preprocessing as skpreprocessing
import datasets
#from decaf.scripts.imagenet import DecafNet
import os


def get_fprop_fn(variable_shape=False, include_pool=True):
    """
    build a theano function that use SAE weights to get convolved(or pooled if
    include_pool is True) features from a given input
    """
    conf = utils.get_config()
    paths = utils.get_paths()
    ae = serial.load(paths['sae']['model'])
    cnn_layer = 'cnn_layer_%i' % (conf['cnn_layers'])
    batch_size = conf[cnn_layer]['batch_size']
    nhid = conf['sae']['nhid']
    patch_size = conf['patch_size']
    region_size = conf['region_size']

    input = T.tensor4('input')
    filter_shape = (nhid, 1, patch_size, patch_size)
    filters = theano.shared(ae.get_weights().T.reshape(filter_shape))

    if variable_shape:
        out = conv.conv2d(input, filters)
    else:
        image_shape = [batch_size, 1, region_size, region_size]
        out = conv.conv2d(input, filters, filter_shape=filter_shape,
                          image_shape=image_shape)

    if include_pool:
        pool_fn = getattr(out, conf['pool_fn'])
        out = pool_fn(axis=(2, 3))
    return theano.function([input], out)


def get_feats_from_ds(ds, conv, preprocess=True):
    conf = utils.get_config()
    cnn_layer = 'cnn_layer_%i' % (conf['cnn_layers'])
    batch_size = conf[cnn_layer]['batch_size']
    patch_size = conf['patch_size']
    nsamples = ds.X.shape[0]
    if preprocess:
        utils.get_pipeline(
            ds.X_topo_space.shape, patch_size, None).apply(ds)

    feats = np.zeros((nsamples, conf['sae']['nhid']))
    for start in range(0, nsamples, batch_size):
        end = min(nsamples, start + batch_size)
        out = conv(ds.get_topological_view()[start:end].transpose(0, 3, 1, 2))
        feats[start:end] = out

    return feats


def get_feats_from_rows(rows, conv, stride):
    """
    Extract features from a list of images using conv function.
    """
    buffer, X = [], []
    iterator = enumerate(rows)
    i, row = iterator.next()
    origin, im = utils.extract_roi(row, stride)
    I = np.asarray(im, dtype=np.float32)
    buffer.append(I)
    last_shape = buffer[0].shape
    for i, row in iterator:
        origin, im = utils.extract_roi(row, stride)
        I = np.asarray(im, dtype=np.float32)
        if last_shape == I.shape:
            buffer.append(I)
        else:
            log.debug('Flushing images with shape: %s' % (str(last_shape)))
            X.extend(get_feats_from_ds(utils.array_to_ds(buffer), conv))
            buffer = [I]
            last_shape = I.shape
    # Flush last buffer
    if len(buffer) > 0:
        X.extend(get_feats_from_ds(utils.array_to_ds(buffer), conv))
    return np.asarray(X)


def get_feats_in_partitions():
    """
    Extracts features from all dataset and split them in train validation and
    test sets
    """
    conf = utils.get_config()
    paths = utils.get_paths()
    rows = utils.load_csv()
    filters = conf['filters']
    region_size = conf['region_size']
    region_stride = conf['region_stride']

    filtered_rows = [
        row for row in rows if utils.check_filter(row, conf['filters'])]
    train_rows, valid_rows, test_rows = utils.split_dataset(
        filtered_rows, conf['valid_percent'], conf['test_percent'], rng=conf['rng_seed'])

    conv = get_fprop_fn(False)
    print 'Getting features from train...'
    X_train = get_feats_from_rows(
        train_rows, conv, conf['stride'])
    print 'Getting features from valid...'
    X_valid = get_feats_from_rows(
        valid_rows, conv, conf['stride'])
    print 'Getting features from test...'
    X_test = get_feats_from_rows(
        test_rows, conv, conf['stride'])
    y_train = [row['classification'] == 'Malign' for row in train_rows]
    y_valid = [row['classification'] == 'Malign' for row in valid_rows]
    y_test = [row['classification'] == 'Malign' for row in test_rows]
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def get_feats_from_oversampled():
    log.info('extracting features...')
    conv = get_fprop_fn()
    ds = datasets.BCDR_On_Memory('train')
    X_train = get_feats_from_ds(ds, conv, preprocess=False)
    y_train = ds.y[:, 0]
    ds = datasets.BCDR_On_Memory('valid')
    X_valid = get_feats_from_ds(ds, conv, preprocess=False)
    y_valid = ds.y[:, 0]
    ds = datasets.BCDR_On_Memory('test')
    X_test = get_feats_from_ds(ds, conv, preprocess=False)
    y_test = ds.y[:, 0]
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def get_feats_from_csv_in_partitions():
    """
    Extract the original features that are distributed in the dataset. Features
    are splitted according with the config.yaml file.
    """
    conf = utils.get_config()
    rows = [row for row in utils.load_csv() if utils.check_filter(row, conf['filters'])]
    train_rows, valid_rows, test_rows = utils.split_dataset(
        rows, conf['valid_percent'], conf['test_percent'], rng=conf['rng_seed'])
    X_train, y_train, X_valid, y_valid, X_test, y_test = [], [], [], [], [], []
    prefixes = ['t_', 'i_', 's_']  # Feature names' prefixes
    datasets = [(X_train, y_train, train_rows),
                (X_test, y_test, test_rows), (X_valid, y_valid, valid_rows)]
    out = []
    for X, y, rows in datasets:
        for row in rows:
            X.append(
                [float(v) for k, v in row.iteritems() if len(filter(k.startswith, prefixes)) > 0])
            y.append(int(row['classification'] == 'Malign'))
        out.extend((np.asarray(X), np.asarray(y)))
    return out


def get_feats_from_csv(rows, prefixes=None):
    """
    Extract the original features that are distributed in the dataset. Features
    are splitted according with the config.yaml file.
    """
    conf = utils.get_config()
    if prefixes == None:
        prefixes = ['t_', 'i_', 's_']  # Feature names' prefixes
    X, y = [], []
    for row in rows:
        X.append(
            [float(v) for k, v in row.iteritems() if len(filter(k.startswith, prefixes)) > 0])
        y.append(
            int(row[conf['csv_class_column']] == conf['csv_positive_class']))
    return (np.asarray(X), np.asarray(y))


def get_all_feats():
    rows = utils.get_filtered_rows()
    y = np.asarray([utils.is_positive(r) for r in rows])
    conv = get_fprop_fn()
    X = get_feats_from_rows(rows, conv, 30)
    return X, y


def get_feats_from_imagenet_in_partitions():
    conf = utils.get_config()
    imagenet_data = os.path.join(
        conf['models_path'], 'decafnet', 'imagenet.decafnet.epoch90')
    imagenet_meta = os.path.join(
        conf['models_path'], 'decafnet', 'imagenet.decafnet.meta')
    net = DecafNet(imagenet_data, imagenet_meta)
    rows = utils.get_filtered_rows()
    sets = utils.split_dataset(
        rows, conf['valid_percent'], conf['test_percent'], rng=conf['rng_seed'])
    feats = []
    ys = []
    for s in sets:
        X = np.zeros((len(s), 4096))
        y = np.zeros(len(s))
        for i, row in enumerate(s):
            try:
                log.info('processing %i-th of %i' % (i, len(s)))
                origin, im = utils.extract_roi(row, 30, True)
                scores = net.classify(np.asarray(im), center_only=True)
                X[i] = net.feature('fc7_cudanet_out')
                y[i] = utils.is_positive(row)
            except:
                continue
        feats.append(X)
        ys.append(y)

    return feats[0], ys[0], feats[1], ys[1], feats[2], ys[2]


def get_feats_from_imagenet(rows):
    conf = utils.get_config()
    imagenet_data = os.path.join(
        conf['models_path'], 'decafnet', 'imagenet.decafnet.epoch90')
    imagenet_meta = os.path.join(
        conf['models_path'], 'decafnet', 'imagenet.decafnet.meta')
    net = DecafNet(imagenet_data, imagenet_meta)
    X = np.zeros((len(rows), 4096))
    y = np.zeros(len(rows))
    for i, row in enumerate(rows):
        try:
            log.info('processing %i-th of %i' % (i, len(rows)))
            origin, im = utils.extract_roi(row, 30, True)
            scores = net.classify(np.asarray(im), center_only=True)
            X[i] = net.feature('fc7_cudanet_out')
            y[i] = utils.is_positive(row)
        except:
            continue

    return X, y


def get_feats_from_cnn(rows, model=None):
    """
    fprop rows using best trained model and returns activations of the
    penultimate layer
    """
    conf = utils.get_config()
    patch_size = conf['patch_size']
    region_size = conf['region_size']
    batch_size = None
    preds = utils.get_predictor(model=model, return_all=True)
    y = np.zeros(len(rows))
    samples = np.zeros(
        (len(rows), region_size, region_size, 1), dtype=np.float32)
    for i, row in enumerate(rows):
        print 'processing %i-th image: %s' % (i, row['image_filename'])
        try:
            samples[i] = utils.get_samples_from_image(row, False)[0]
        except ValueError as e:
            print '{1} Value error: {0}'.format(str(e), row['image_filename'])
        y[i] = utils.is_positive(row)
    ds = DenseDesignMatrix(topo_view=samples)
    pipeline = utils.get_pipeline(
        ds.X_topo_space.shape, patch_size, batch_size)
    pipeline.apply(ds)
    return preds[-2](ds.get_topological_view()), y
