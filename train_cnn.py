#!/usr/bin/env python

from pylearn2.utils import serial
from pylearn2.config import yaml_parse
import utils
import datasets
import os
import sys


def main():
    conf_file = sys.argv[1] if len(sys.argv) > 1 else None
    conf = utils.get_config(conf_file)
    paths = utils.get_paths(conf)
    cnn_layer = 'cnn_layer_%i' % (conf['cnn_layers'])

    with open(paths[cnn_layer]['yaml']) as f:
        src = f.read()

    # Get batch size from validation set to report roc_auc from a single batch
    ds = datasets.BCDR('valid')
    monitoring_batch_size = ds.y.shape[0]
    ds.h5file.close()
    train_ds_class = 'BCDR_On_Memory' if conf['train_on_memory'] else 'BCDR'
    valid_ds_class = 'BCDR_On_Memory' if conf['valid_on_memory'] else 'BCDR'

    params = utils.flatten(conf)

    params.update({
        'train_ds_class': train_ds_class,
        'valid_ds_class': valid_ds_class,
        'monitoring_batch_size': monitoring_batch_size,
        'save_path': paths[cnn_layer]['model'],
        'best_path': paths[cnn_layer]['best_model']
    })
    train = yaml_parse.load(src % params)
    if os.path.isfile(train.save_path):
        print('%s already exists, skipping...' % (train.save_path))
    else:
        if conf['load_pretrained']:
            print('Setting pretrained filters...')
            ae = serial.load(paths['sae']['model'])

            W = ae.get_weights().T.reshape(
                train.model.layers[0].transformer._filters_shape)
            train.model.layers[0].transformer._filters.set_value(W)

        train.main_loop()
    print('Done!')

if __name__ == '__main__':
    main()
