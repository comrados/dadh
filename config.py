import warnings
import torch


class Default(object):
    load_model_path = None  # load model path

    pretrain_model_path = './data/imagenet-vgg-f.mat'

    # visualization
    vis_env = 'main'  # visdom env
    vis_env = False
    vis_port = 8097  # visdom port
    flag = 'mir'

    batch_size = 128
    image_dim = 4096
    hidden_dim = 4096
    modals = 2
    valid = True  # whether to use validation
    valid_freq = 1
    max_epoch = 300

    bit = 64  # hash code length
    lr = 0.0001  # initial learning rate

    device = 'cuda:0'
    # device = 'cpu'

    # hyper-parameters
    alpha = 10
    gamma = 10
    beta = 1
    mu = 0.00001
    lamb = 1

    margin = 0.4
    dropout = True

    proc = None

    def data(self, flag):
        if flag == 'mir':
            self.dataset = 'flickr25k'
            self.data_path = r'/media/george/Data/RS data/FLICKR-25K.mat'
            self.db_size = 18015
            self.num_label = 24
            self.query_size = 2000
            self.text_dim = 1386
            self.training_size = 10000
        if flag == 'nus':
            self.dataset = 'nus-wide'
            self.data_path = './data/NUS-WIDE-TC21.mat'
            self.db_size = 193734
            self.num_label = 21
            self.query_size = 2100
            self.text_dim = 1000
            self.training_size = 10000
        if flag == 'ucm':
            self.dataset = 'ucm'
            self.data_path = './data/dataset_UCM_with_embeddings_reduced.h5'
            self.db_size = 9450
            self.num_label = 21
            self.query_size = 1050
            self.text_dim = 266
            self.training_size = 9450

    def parse(self, kwargs):
        """
        update configuration by kwargs.
        """
        for k, v in kwargs.items():
            if k == 'flag':
                self.data(v)
            if k == 'proc':
                self.proc = v
            if k == 'device':
                self.device = v
            if not hasattr(self, k):
                warnings.warn("Warning: opt has no attribute %s" % k)
            setattr(self, k, v)

        print('Configuration:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and str(k) != 'parse' and str(k) != 'data':
                print('\t{0}: {1}'.format(k, getattr(self, k)))


opt = Default()
