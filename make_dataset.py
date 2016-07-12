import utils
from pylearn2.utils import serial
import h5py
import numpy as np
import sys

if __name__ == "__main__":
    conf_file = sys.argv[1] if len(sys.argv) > 1 else None
    conf = utils.get_config(conf_file)
    paths = utils.get_paths()
    region_size = conf['region_size']
    region_stride = conf['region_stride']

    train_rows, valid_rows, test_rows = utils.split_dataset(
        utils.get_filtered_rows(), conf['valid_percent'],
        conf['test_percent'], rng=conf['rng_seed'])

    rowsdict = {'train': train_rows, 'valid': valid_rows, 'test': test_rows}
    nsamples = {}

    prefixes = ['s_', 'i_', 't_']  # Feature names' prefixes
    for subset, subrows in rowsdict.iteritems():
        X = None
        y = []
        feats = []
        for row in subrows:
            samples = utils.get_samples_from_image(
                row, oversampling=(subset == 'train' and conf['oversampling']))
            print "%i samples to %s taken from %s" % (
                len(samples), subset, row['image_filename'])
            if len(samples) == 0:
                continue
            samples = np.array(samples, dtype=np.float32) / 255.0
            # linearized dimension of im
            ndim = np.cumprod(samples.shape[1:])[-1]
            samples = samples.reshape(samples.shape[0], ndim)
            if X is None:
                f = h5py.File(paths['raw_' + subset], 'w')
                X = f.create_dataset('X', (0, ndim), maxshape=(None, ndim),
                                     compression = "gzip", compression_opts = 9)

            X.resize(X.shape[0] + samples.shape[0], axis=0)
            X[-len(samples):] = samples
            y.extend([utils.is_positive(row) for i in range(len(samples))])
            feats.extend(
                [[float(v) for k, v in row.iteritems() if len(filter(k.startswith, prefixes)) > 0]] * samples.shape[0])

        y = np.asarray(y)
        y = np.vstack((y, 1 - y)).T
        f.create_dataset('y', data=y)
        f.create_dataset('feats', data=np.asarray(feats))
        nsamples[subset] = X.shape[0]
        f.close()
        serial.save(paths[subset + '_rows'], subrows)
