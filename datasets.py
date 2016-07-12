from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets.vector_spaces_dataset import VectorSpacesDataset
from pylearn2.space import VectorSpace, CompositeSpace, Conv2DSpace
import tables
import theano
import utils
from pylearn2.utils import string_utils as string


class BCDR(dense_design_matrix.DenseDesignMatrixPyTables):

    def __init__(self, which_set):
        conf = utils.get_config()
        paths = utils.get_paths()
        region_size = conf['region_size']
        self.h5file = tables.open_file(paths[which_set])
        node = self.h5file.root.Data
        axes = ('b', 0, 1, 'c')
        channels = node.X.shape[1] / (region_size * region_size)
        view_converter = dense_design_matrix.DefaultViewConverter(
            (region_size, region_size, channels), axes)
        super(BCDR, self).__init__(
            X=node.X, view_converter=view_converter, y=node.y)


class BCDR_On_Memory(dense_design_matrix.DenseDesignMatrix):

    def __init__(self, which_set):
        conf = utils.get_config()
        paths = utils.get_paths()
        region_size = conf['region_size']
        h5file = tables.open_file(paths[which_set])
        node = h5file.root.Data
        X = node.X.read()
        channels = node.X.shape[1] / (region_size * region_size)
        X = X.reshape(
            (X.shape[0], conf['region_size'], conf['region_size'], channels))
        y = node.y.read()
        h5file.close()
        super(BCDR_On_Memory, self).__init__(topo_view=X, y=y)


class BCDRComposite(VectorSpacesDataset):

    def __init__(self, which_set, train_dataset=None):
        conf = utils.get_config()
        paths = utils.get_paths()
        region_size = conf['region_size']
        h5file = tables.open_file(paths[which_set])
        node = h5file.root.Data
        X = node.X.read()
        num_channels = node.X.shape[1] / (region_size * region_size)
        im_shape = (conf['region_size'], conf['region_size'])
        X = X.reshape((X.shape[0],) + im_shape + (num_channels,))
        y = node.y.read()
        self.feats = node.feats.read()
        h5file.close()
        if train_dataset is None:
            self.feats_mean = self.feats.mean(axis=0)
            self.feats_std = self.feats.std(axis=0)
            self.feats = (self.feats - self.feats_mean) / self.feats_std
        else:
            feats_mean = train_dataset.feats_mean
            feats_std = train_dataset.feats_std
            self.feats = (self.feats - feats_mean) / feats_std

        self.y = self.feats
        source = ('features', 'targets0', 'targets1')
        conv_space = Conv2DSpace(
            shape=im_shape, num_channels=num_channels, axes=('b', 0, 1, 'c'))
        target_space = VectorSpace(y.shape[1])
        shape_space = VectorSpace(self.feats.shape[1])

        space = CompositeSpace([conv_space, target_space, shape_space])

        data = (X.astype(theano.config.floatX),
                y.astype(theano.config.floatX),
                self.feats.astype(theano.config.floatX),)

        super(BCDRComposite, self).__init__(
            data=data, data_specs=(space, source))
