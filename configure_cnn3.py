import os
import sys
import yaml
import numpy as np
from bcdr import utils
from pylearn2.utils import serial

num_jobs = 25
rng = np.random.RandomState([2014, 12, 22])
conf_file = sys.argv[1] if len(sys.argv) > 1 else None
conf = utils.get_config(conf_file)
model_name = conf['model_name']


def random_init_string(sparse=False):
    if rng.randint(2) and sparse:
        sparse_init = rng.randint(10, 30)
        return "sparse_init: " + str(sparse_init)
    irange = 10. ** rng.uniform(-2.3, -1.)
    return "irange: " + str(irange)


def rectifier_bias():
    if rng.randint(2):
        return 0
    return rng.uniform(0, .3)

for job_id in xrange(num_jobs):
    conf['model_name'] = '%s_%02i' % (model_name, job_id)
    paths = utils.get_paths(conf)
    if os.path.isfile(paths['cnn_layer_3']['model_yaml']):
        print '%s already exists, skipping...' % (paths['cnn_layer_3']['model_yaml'])
        continue

    # Initialization params
    conf['h0']['init'] = random_init_string()
    conf['h0']['bias'] = rectifier_bias()
    conf['h1']['init'] = random_init_string()
    conf['h1']['bias'] = rectifier_bias()
    conf['hfc']['init'] = random_init_string(sparse=True)
    conf['y']['init'] = random_init_string(sparse=True)

    # Regularization params
    # http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf: Typical
    # values of c range from 3 to 4
    conf['h0']['col_norm'] = rng.uniform(1., 5.)
    conf['h1']['col_norm'] = rng.uniform(1., 5.)
    conf['hfc']['col_norm'] = rng.uniform(1., 5.)
    conf['y']['col_norm'] = rng.uniform(1., 5.)

    # SGD params
    conf['learning_rate'] = 10. ** rng.uniform(-3., -1.)
    conf['lr_sat'] = rng.randint(50, 100)
    conf['lr_decay'] = 10. ** rng.uniform(-3, -1)
    conf['final_momentum'] = rng.uniform(.7, .99)
    if rng.randint(2):
        conf['msat'] = 2
    else:
        conf['msat'] = rng.randint(50, 100)

    with open(paths['cnn_layer_3']['model_yaml'], 'w') as f:
        yaml.dump(conf, f)
