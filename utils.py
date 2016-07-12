# -*- coding: utf-8 -*-

import csv
from PIL import Image, ImageDraw
from scipy import ndimage
import os.path
import numpy as np
import tables
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc
from pylearn2.utils.string_utils import preprocess
from pylearn2.utils import serial
import pylab
import theano
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.preprocessing import ExtractGridPatches
from pylearn2.datasets import preprocessing

import yaml
from pylearn2.utils.rng import make_np_rng
import logging as log
import collections
import matplotlib.pyplot as plt


config = None


def get_config(filename=None, reload=False):
    global config
    if config is not None and not reload:
        return config
    if filename is None:
        raise Exception(
            'Configuration has not been loaded previously, filename parameter is required')
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        filename = os.path.join(curr_dir, 'config.yaml')
    with open(filename) as f:
        config = yaml.load(f)
    return config


def get_paths(conf=None):
    if conf is None:
        conf = get_config()
    paths = {}
    data_path = os.path.join('data', conf['ds_dir'])
    paths['data_path'] = data_path
    data_path = os.path.join(data_path, 'cv-sets', conf['ds_name'])
    paths['bcdr'] = os.path.dirname(os.path.realpath(__file__))
    prefix = 'rs%i_rt%i_ov%0.2e' % (
        conf['region_size'], conf['region_stride'], conf['min_overlap'])
    paths['raw_train'] = os.path.join(data_path, prefix + '_train.hdf5')
    paths['raw_valid'] = os.path.join(data_path, prefix + '_valid.hdf5')
    paths['raw_test'] = os.path.join(data_path, prefix + '_test.hdf5')
    paths['train_rows'] = os.path.join(data_path, prefix + '_train_rows.pkl')
    paths['valid_rows'] = os.path.join(data_path, prefix + '_valid_rows.pkl')
    paths['test_rows'] = os.path.join(data_path, prefix + '_test_rows.pkl')
    paths['preprocessor'] = os.path.join(
        data_path, prefix + '_preprocessor.pkl')
    data_path = os.path.join(data_path, 'normalized')
    paths['train'] = os.path.join(data_path, prefix + '_train.hdf5')
    paths['valid'] = os.path.join(data_path, prefix + '_valid.hdf5')
    paths['test'] = os.path.join(data_path, prefix + '_test.hdf5')

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    paths['models_path'] = os.path.join(conf['models_path'], conf['ds_name'])
    # sparse autoencoders paths
    sae = {}
    sae['yaml'] = os.path.join(paths['bcdr'], 'yaml', 'sae.yaml')
    prefix = '%i_ps%i' % (conf['region_size'], conf['patch_size'])
    sae['train'] = os.path.join(data_path, 'patches', prefix + '_train.hdf5')
    sae['valid'] = os.path.join(data_path, 'patches', prefix + '_valid.hdf5')
    sae['model'] = os.path.join(
        paths['models_path'], 'sae', conf['model_name'] + '.pkl')
    paths['sae'] = sae

    if not os.path.exists(os.path.dirname(sae['train'])):
        os.makedirs(os.path.dirname(sae['train']))
    if not os.path.exists(os.path.dirname(sae['model'])):
        os.makedirs(os.path.dirname(sae['model']))

    # CNN x-layer paths
    cnn_layer = 'cnn_layer_%i' % (conf['cnn_layers'])
    cnn = {}
    cnn['yaml'] = os.path.join(paths['bcdr'], 'yaml', cnn_layer + '.yaml')
    prefix = conf['model_name']
    cnn['model'] = os.path.join(
        paths['models_path'], 'cnn', 'layer_%i' % (conf['cnn_layers']), prefix + '.pkl')
    cnn['model_yaml'] = os.path.join(
        paths['models_path'], 'cnn', 'layer_%i' % (conf['cnn_layers']), prefix + '.yaml')
    cnn['best_model'] = os.path.join(
        paths['models_path'], 'cnn', 'layer_%i' % (conf['cnn_layers']), prefix + '_best.pkl')
    paths[cnn_layer] = cnn

    return paths


def export_rois_to_files(rows):
    for row in rows:
        fname = row['image_filename']
        im = get_roi(row)
        dirname = 'rois/' + os.path.dirname(fname)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        im.save('rois/' + fname)


def get_roi(row):
    fname = row['image_filename']
    paths = get_paths()
    im = Image.open(os.path.join(paths['data_path'], fname))
    x, y = get_points(row)
    return im.crop((min(x), min(y), max(x), max(y)))


def dump_dataset(rows, filename):
    '''
    Build a hdf5 file with pylearn2 format. Force images to have the same size
    by either cropping in the center or filling with black pixels
    '''
    conf = get_config()
    nb_samples = len(rows)
    nb_classes = 2
    X = np.zeros((nb_samples, conf['img_height'], conf['img_width']))
    y = np.zeros((nb_samples, nb_classes))
    queue = []
    for i, row in enumerate(rows):
        fname = row['image_filename']
        paths = get_paths()
        im = Image.open(os.path.join(paths['data_path'], fname))
        # Make sure images have the same size
        if im.size != (conf['img_width'], conf['img_height']):
            M = np.asarray(im)
            offsetX = min(conf['img_height'], im.size[1])
            offsetY = min(conf['img_width'], im.size[0])
            startX = (conf['img_height '] - offsetX) / 2
            startY = (conf['img_width '] - offsetY) / 2
            X[i, startX:offsetX + startX, startY:offsetY +
                startY] = M[0:offsetX, 0:offsetY]
        else:
            X[i] = np.asarray(im)
        y[i, int(is_positive(row))] = 1

    create_hdf5_file(X, y, filename)


def create_hdf5_file(X, y, filename):
    f = h5py.File(filename, "w")
    X = X.reshape(X.shape + (1,))  # Include channel dimension
    f.create_dataset('X', data=X)
    f.create_dataset('y', data=y)
    f.close()


def _load_csv(file=None):
    conf = get_config()
    if file is None:
        file = conf['csv_file']
    path = os.path.join(get_paths()['data_path'], file)
    with open(path) as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        return [dict(zip(headers, map(str.strip, row))) for row in reader]

def load_csv():
    conf = get_config()
    return [dict(r[0],  **r[1]) for r in zip(_load_csv(conf['csv_features_file']), _load_csv())]

def get_filtered_rows():
    conf = get_config()
    return [r for r in load_csv() if check_filter(r, conf['filters'])]


def rows_to_patients(rows):
    patients = {}
    for row in rows:
        patient_id = row['patient_id']
        label = is_positive(row)
        patients[patient_id] = label
    return patients


def patients_to_rows(patients, rows):
    return [row for row in rows if row['patient_id'] in patients]


def split_patients(patients, valid_percent, test_percent, rng=(2014, 10, 22)):
    if isinstance(rng, (list, tuple)):
        rng = make_np_rng(None, rng, which_method='uniform')

    vals = np.asarray(patients.values())
    keys = np.asarray(patients.keys())
    sss = StratifiedShuffleSplit(
        vals, n_iter=1, test_size=test_percent, random_state=rng)
    remaining_idx, test_idx = sss.__iter__().next()

    if valid_percent > 0:
        # Rate of samples required to build validation set
        valid_rate = valid_percent / (1 - test_percent)

        sss = StratifiedShuffleSplit(
            vals[remaining_idx], n_iter=1, test_size=valid_rate, random_state=rng)
        tr_idx, val_idx = sss.__iter__().next()
        valid_idx = remaining_idx[val_idx]
        train_idx = remaining_idx[tr_idx]
    else:
        train_idx = remaining_idx
        valid_idx = []

    train_patients = dict(zip(keys[train_idx], vals[train_idx]))
    valid_patients = dict(zip(keys[valid_idx], vals[valid_idx]))
    test_patients = dict(zip(keys[test_idx], vals[test_idx]))
    return train_patients, valid_patients, test_patients


def split_dataset(rows, valid_percent, test_percent, rng=(2014, 10, 22), patients=None):
    if patients is None:
        patients = rows_to_patients(rows)
    train_patients, valid_patients, test_patients = split_patients(
        patients, valid_percent, test_percent, rng)
    train_rows = patients_to_rows(train_patients, rows)
    valid_rows = patients_to_rows(valid_patients, rows)
    test_rows = patients_to_rows(test_patients, rows)
    return (train_rows, valid_rows, test_rows)


def check_filter(row, filter):
    for key, value in filter.iteritems():
        if str(row[key]) not in value:
            return False
    return True


def get_outline(row, origin, scale_factor=1.0, convex_hull=False):
    """
    returns a polygon object describing the outline. The output is relative to
    the origin param.
    """
    from shapely.geometry import Point, Polygon
    from shapely import affinity
    x, y = get_points(row)
    # Close the polygon
    x.append(x[0])
    y.append(y[0])
    poly = affinity.translate(Polygon(zip(x, y)), -origin[0], -origin[1])
    x, y = poly.exterior.coords.xy

    opts = np.asarray([1 + i * (10.0 ** -j)
                       for i in range(1, 10) for j in range(1, 12)])
    opts.sort()
    opts = np.insert(opts, 0, 1)  # Try '1' first

    if convex_hull:
        outline = Polygon(zip(x, y)).convex_hull
    else:
        outline = None

    for eps in opts:
        if isinstance(outline, Polygon):
            break
        outline = Polygon(zip(x, y)).buffer(eps)

    if not isinstance(outline, Polygon):
        log.warning(
            'Cannot find a valid region for %s. Using convex hull' % (row['image_filename']))
        outline = Polygon(zip(x, y)).convex_hull

    if scale_factor < 1.0:
        return affinity.scale(outline, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
    else:
        return outline


def get_mask(I, outline):
    mask = Image.new('L', I.T.shape)
    draw = ImageDraw.Draw(mask)
    points = zip(outline.exterior.xy[0], outline.exterior.xy[1])
    draw.polygon(points, fill=1)
    mask = np.asarray(mask)
    return ndimage.morphology.distance_transform_edt(mask)


def get_samples_from_image(row, oversampling):
    '''
    Sample patches from an image.
    '''
    from shapely.geometry import box
    fname = row['image_filename']
    paths = get_paths()
    conf = get_config()
    origin = (0, 0)
    scale_factor = 1.0
    #plot_image(row, filename='%s_0.png' % (os.path.basename(fname)))
    try:
        if conf['crop_image']:
            origin, im = extract_roi(
                row, conf['region_stride'], squared=conf['scale_image'])
        else:
            im = Image.open(os.path.join(paths['data_path'], fname))
    except Exception as e:
        log.warn('%s could not be processed %s.' %
                 (os.path.join(paths['data_path'], fname), str(e)))
        return []
    im = im.convert('L')
    #im.save('%s_1.png' % (os.path.basename(fname)))

    if conf['scale_image'] and conf['region_size'] != im.size[0]:
        scale_factor = float(conf['region_size']) / float(im.size[0])
        im = im.resize((conf['region_size'], conf['region_size']))
        #im.save('%s_2.png' % (os.path.basename(fname)))
    outline = get_outline(row, origin, scale_factor, conf['convex_hull'])
    #plot_image(row, im.copy(), False, '%s_3.png' % (os.path.basename(fname)), [outline])
    I = np.asarray(im, dtype=np.float32) / 255.0

    if conf['preprocessing']['per_roi']:
        ds = array_to_ds(I.reshape((1,) + I.shape))
        get_pipeline(I.shape, conf['patch_size'], None).apply(ds)
        I = ds.X.reshape(I.shape)
        #plot_image(row, array_to_im(I).copy(), False, '%s_4.png' % (os.path.basename(fname)))
        #plot_image(row, array_to_im(I).copy(), False, '%s_5.png' % (os.path.basename(fname)), [outline])

    if conf['apply_mask']:
        # I = np.dstack((I, get_mask(I, outline))) #Add as a new channel
        # I = (get_mask(I, outline) * I).reshape(I.shape + (1,)) #apply on raw
        I = get_mask(I, outline).reshape(I.shape + (1,))  # Return only mask
    else:
        I = I.reshape(I.shape + (1,))

    outline_area = outline.area
    bounds = outline.bounds
    samples = []
    coords = []
    max_overlap = 0
    min_overlap = conf['min_overlap']

    im_size = I.shape[:2][::-1]  # convert array to im coords
    for i, patch in enumerate(create_grid(bounds, im_size)):
        x21, y21, x22, y22 = patch
        window = box(x21, y21, x22, y22)
#        plot_image(row, array_to_im(I.squeeze()), True, '%s_p%i.png' %
#                   (os.path.basename(fname), i), [window, outline])
        overlap = outline.intersection(window).area
        max_overlap = max(max_overlap, overlap)
        if overlap / outline_area > min_overlap or overlap / window.area > min_overlap:
            region = I[y21:y22, x21:x22, :]
            if oversampling:
                samples.extend([np.asarray(r) for r in oversample(region)])
            else:
                samples.append(np.asarray(region))
            coords.append((x21, y21, x22, y22))
    if len(samples) == 0:
        log.warning('Max Overlap: %f, %f' %
                    (max_overlap / window.area, max_overlap / outline_area))
    return samples


def create_grid(bounds, im_size):
    conf = get_config()
    step_x = conf['region_size']
    step_y = conf['region_size']
    stride_x = conf['region_stride']
    stride_y = conf['region_stride']
    start_x = int(max(0, bounds[0] - step_x + stride_x))
    start_y = int(max(0, bounds[1] - step_y + stride_y))
    end_x = int(min(im_size[0], bounds[2] + step_x))
    end_y = int(min(im_size[1], bounds[3] + step_y))
    x = np.arange(start_x, end_x - step_x + stride_x, stride_x)
    y = np.arange(start_y, end_y - step_y + stride_y, stride_y)
    x, y = np.meshgrid(x, y)
    return [(i, j, i + step_x, j + step_y) for i, j in zip(x.flatten(), y.flatten())]


def oversample(I):
    samples = []
    samples.append(I)
    samples.append(ndimage.rotate(I, 90))
    samples.append(ndimage.rotate(I, 180))
    samples.append(ndimage.rotate(I, 270))
    samples.append(np.fliplr(I))
    samples.append(np.flipud(I))
    samples.append(np.fliplr(samples[1]))
    samples.append(np.flipud(samples[1]))
    return samples


def plot_image(row, im=None, outlined=True, filename=None, other_polygons=[]):
    """
    Plot an image. if im is given it will not load the image from file.
    """
    fname = row['image_filename']
    paths = get_paths()
    if im is None:
        im = Image.open(os.path.join(paths['data_path'], fname))
    im = im.convert('RGB')
    width, height = im.size
    draw = ImageDraw.Draw(im)

    if outlined:
        color = 'red' if is_positive(row) else 'green'
        x, y = get_points(row)
        points = zip(x, y)
        points.append(points[0])
        draw.line(points, fill=color, width=4)

    for pgn in other_polygons:
        color = 'blue'
        points = zip(pgn.exterior.xy[0], pgn.exterior.xy[1])
        draw.line(points, fill=color, width=2)

    if filename:
        im.save(filename)


def init_hdf5(path, shapes, title="Pylearn2 Dataset", filters=None):
    from theano import config
    if filters is None:
        filters = tables.Filters(complib='blosc', complevel=5)
    x_shape, y_shape, feats_shape = shapes
    h5file = tables.open_file(path, mode="w", title=title)
    node = h5file.create_group(h5file.root, "Data", "Data")
    atom = (tables.Float32Atom() if config.floatX == 'float32'
            else tables.Float64Atom())
    h5file.create_carray(node, 'X', atom=atom, shape=x_shape,
                         title="Data values", filters=filters)
    h5file.create_carray(node, 'y', atom=atom, shape=y_shape,
                         title="Data targets", filters=filters)
    h5file.create_carray(node, 'feats', atom=atom, shape=feats_shape,
                         title="Data targets", filters=filters)
    return h5file, node


def h5py_to_tables(inputfile, outputfile, title='exported_pytables'):
    hf = tables.open_file(inputfile)
    shapes = (hf.root.X.shape, hf.root.y.shape, hf.root.feats.shape)
    h5file, node = init_hdf5(outputfile, shapes, title)
    for i, x in enumerate(hf.root.X):
        node.X[i] = x

    for i, y in enumerate(hf.root.y):
        node.y[i] = y

    for i, feats in enumerate(hf.root.feats):
        node.feats[i] = feats
    hf.close()
    return h5file, node


def extract_roi(row, stride, squared=False):
    """
    Extracts ROI of an image. Returns the top-left coords and the cropped image
    """
    conf = get_config()
    paths = get_paths()
    fname = row['image_filename']
    im = Image.open(os.path.join(paths['data_path'], fname))
    img_width, img_height = im.size
    x, y = get_points(row)
    if squared:
        max_axis = max(max(x) - min(x), max(y) - min(y))
        width = max(max_axis, conf['region_size'])
        height = max(max_axis, conf['region_size'])
    else:
        width = max(max(x) - min(x), conf['region_size'])
        height = max(max(y) - min(y), conf['region_size'])
    if width % stride > 0:
        width = width + stride - (width % stride)
    if height % stride > 0:
        height = height + stride - (height % stride)
    center_x = min(x) + (max(x) - min(x) - width) / 2
    center_y = min(y) + (max(y) - min(y) - height) / 2
    start_x = min(max(center_x, 0), img_width - width)
    start_y = min(max(center_y, 0), img_height - height)

    im = im.crop((start_x, start_y, start_x + width, start_y + height))
    return (start_x, start_y), im


def get_points(row):
    conf = get_config()
    x = map(int, row['lw_x_points'].split(conf['csv_point_separator']))
    y = map(int, row['lw_y_points'].split(conf['csv_point_separator']))
    return (x, y)


def is_positive(row):
    conf = get_config()
    return row[conf['csv_class_column']] == conf['csv_positive_class']


def get_predictor(model=None, return_all=False):
    if model is None:
        # Load config and read params
        conf = get_config()
        paths = get_paths()
        cnn_layer = 'cnn_layer_%i' % (conf['cnn_layers'])
        model = paths[cnn_layer]['best_model']

    if isinstance(model, basestring):
        # Load model and dataset
        model = serial.load(model)

    X = model.get_input_space().make_theano_batch()
    Y = model.fprop(X, return_all=return_all)
    if return_all:
        predict = [theano.function([X], y) for y in Y]
    else:
        predict = theano.function([X], Y)
    return predict


def mat2gray(x):
    return (x - x.min()) / (x.max() - x.min())


def get_pipeline(img_shape, patch_size, batch_size):
    pipeline = preprocessing.Pipeline()
    conf = get_config()
    if conf['preprocessing']['remove_mean']:
        pipeline.items.append(preprocessing.RemoveMean())
    if conf['preprocessing']['gcn']:
        pipeline.items.append(
            preprocessing.GlobalContrastNormalization(batch_size=batch_size)
        )
    if conf['preprocessing']['lcn']:
        # LCN requires uneven patch size
        lcn_patch_size = patch_size + 1 - (patch_size % 2)
        pipeline.items.append(
            preprocessing.LeCunLCN(
                img_shape, kernel_size=lcn_patch_size)
        )
    return pipeline


def array_to_im(X):
    return Image.fromarray(np.uint8(pylab.cm.Greys_r(mat2gray(X)) * 255))


def array_to_ds(X):
    """
    Build a DenseDesignMatrix with topo_view using X.
    X: a nsamples x pixels numpy array, or a list of linearized images
    """
    if type(X) is list:
        X = np.asarray(X)
    return DenseDesignMatrix(topo_view=X.reshape(X.shape + (1,)))


def flatten(d, parent_key='', sep='_'):
    """
    Flatten a dictionary with nested subdictionaries
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def plot_roc_curve(y, score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr["micro"], tpr["micro"], _ = roc_curve(y, score)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.4f})'
             ''.format(roc_auc["micro"]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
