import sys
import os.path
import utils
from pylearn2.utils import serial, string_utils as string
from pylearn2.datasets import preprocessing, dense_design_matrix

if __name__ == "__main__":
    conf_file = sys.argv[1] if len(sys.argv) > 1 else None
    conf = utils.get_config(conf_file)
    paths = utils.get_paths()
    patch_size = conf['patch_size']
    region_size = conf['region_size']
    batch_size = conf['preprocessing']['batch_size']

    h5file, node = utils.h5py_to_tables(
        paths['raw_train'], paths['train'], title=conf['ds_name'])

    axes = ('b', 0, 1, 'c')
    channels = int(node.X.shape[1] / (region_size * region_size))
    view_converter = dense_design_matrix.DefaultViewConverter(
        (region_size, region_size, channels), axes)
    train = dense_design_matrix.DenseDesignMatrixPyTables(
        X=node.X, view_converter=view_converter, y=node.y)
    train.h5file = h5file

    # If dataset was preprocessed by ROI it should not be preprocessed by
    # region
    if conf['preprocessing']['per_roi']:
        pipeline = preprocessing.Pipeline()
    else:
        pipeline = utils.get_pipeline(
            train.X_topo_space.shape, patch_size, batch_size)
    pipeline.items.append(
        preprocessing.ShuffleAndSplit(conf['rng_seed'], 0, node.X.shape[0]))
    train.apply_preprocessor(preprocessor=pipeline, can_fit=True)
    h5file.close()

    pipeline.items.pop()
    h5file, node = utils.h5py_to_tables(
        paths['raw_valid'], paths['valid'], title='BCDR')
    valid = dense_design_matrix.DenseDesignMatrixPyTables(
        X=node.X, view_converter=view_converter, y=node.y)
    valid.h5file = h5file
    valid.apply_preprocessor(preprocessor=pipeline, can_fit=False)
    h5file.close()

    h5file, node = utils.h5py_to_tables(
        paths['raw_test'], paths['test'], title='BCDR')
    test = dense_design_matrix.DenseDesignMatrixPyTables(
        X=node.X, view_converter=view_converter, y=node.y)
    test.h5file = h5file
    test.apply_preprocessor(preprocessor=pipeline, can_fit=False)
    h5file.close()

    serial.save(paths['preprocessor'], pipeline)
