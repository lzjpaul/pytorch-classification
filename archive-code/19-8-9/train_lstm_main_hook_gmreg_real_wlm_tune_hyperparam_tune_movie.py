## what for:
## (1) using "for" for prior_beta, reg_lambda, weight_decay


## references:
## https://github.com/spro/practical-pytorch/tree/master/char-rnn-classification
## refer to
## https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
## https://github.com/spro/practical-pytorch/tree/master/char-rnn-classification
## https://github.com/lzjpaul/pytorch/blob/LDA-regularization/examples/cifar-10-tutorial/mimic_lstm_lda.py
## https://github.com/lzjpaul/pytorch/blob/residual-knowledge-driven/examples/residual-knowledge-driven-example/mlp_residual.py
## https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#sphx-glr-beginner-nlp-sequence-models-tutorial-py
## https://github.com/pytorch/examples/blob/master/word_language_model/main.py
## https://github.com/lzjpaul/pytorch/blob/residual-knowledge-driven/examples/residual-knowledge-driven-example-test-lda-prior/train_lstm_main_hook_resreg_real.py

## learning rate annealing:
# https://blog.csdn.net/u012436149/article/details/70666068 
# https://blog.csdn.net/u012436149/article/details/70666068
#--> only one parameter group
# https://discuss.pytorch.org/t/adaptive-learning-rate/320/3
# https://stackoverflow.com/questions/52660985/pytorch-how-to-get-learning-rate-during-training

# https://pytorch.org/docs/stable/optim.html
# https://gist.github.com/j-min/a07b235877a342a1b4f3461f45cf33b3
# https://discuss.pytorch.org/t/adaptive-learning-rate/320/2
# https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html
# https://discuss.pytorch.org/t/different-learning-rate-for-a-specific-layer/33670

## different from MIMIC-III
## (1) no seq_length
## (2) !! batch_first not True nned to modify many!!! 
##     --> forward function in BasicRNNBlock !! input and output are modified!!
##     --> forward function in ResNetRNN !! input and output and F.sigmoid(self.fc1(x[:, -1, :])) need to be modified!!
##     --> judge batch_first very slow ??
## (3) softmax loss ???


## Attention:
## (1) batch_first set as argument --> for calculating correlation also!!
## (2) label_num
## (3) batch_size are passed to the functions
## (4) different model different params???
## (5) sequence length is not fixed, so I do not pass in sequence_length + init_hidden need to pass in features[0] as batch_size + forward() batchsize need to be 
##     calculated while init_hidden batchsize is passed in by features[0]
## Attenton after TKDE revision:
## (1) for non-wlm, not divide by (labelnum * train * seqnum) yet!!

import torch
import torch.nn as nn
from torch.autograd import Variable
from init_linear import InitLinear
# from res_regularizer import ResRegularizer
import torch.utils.data as Data
import torch.autograd as autograd
import random
import time
import math
import sys
import numpy as np
import scipy.sparse
import logging
import argparse
from torch.optim.adam import Adam
import torch.optim as optim
import datetime
import torch.nn.functional as F
from mimic_metric import *
import data
import gm_prior_optimizer_pytorch


features = []

class BasicRNNBlock(nn.Module):
    # no seq_length ?? batch_fist not true ??
    def __init__(self, gpu_id, input_dim, hidden_dim, batch_first):
        super(BasicRNNBlock, self).__init__()
        self.gpu_id = gpu_id # return init_hidden ...
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, batch_first=batch_first)

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # print ('init BasicRNNBlock')
        if self.gpu_id >= 0:
            return autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda(self.gpu_id))
        else:
            return autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))
    ### ??? !! batch_first: inputs.view(batch_size, -1, self.input_dim), self.hidden)
    ### ??? !! rnn_out return [:, -1, :]
    ## actually, this is useless of the input x is already organized according to batch_first
    def forward(self, x):
        logger = logging.getLogger('gm_reg')
        logger.debug('Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('input size:')
        logger.debug (x.data.size())
        logger.debug ('input norm: %f', x.data.norm())
        if self.batch_first:
            batch_size = x.size()[0]
            rnn_out, self.hidden = self.rnn(
                x.view(batch_size, -1, self.input_dim), self.hidden)
        else:
            batch_size = x.size()[1]
            rnn_out, self.hidden = self.rnn(
                x.view(-1, batch_size, self.input_dim), self.hidden)
        logger.debug ('rnn_out size: ')
        logger.debug (rnn_out.data.size())
        logger.debug ('rnn_out norm: %f', rnn_out.data.norm())
        # print ('rnn_out norm: ', np.linalg.norm(rnn_out.data.cpu().numpy()))
        # print ('rnn_out/x ratio: ', np.linalg.norm(rnn_out.data.cpu().numpy())/np.linalg.norm(x.data.cpu().numpy()))
        return rnn_out

class BasicLSTMBlock(nn.Module):
    # no seq_length ?? batch_fist not true ??
    def __init__(self, gpu_id, input_dim, hidden_dim, batch_first):
        super(BasicLSTMBlock, self).__init__()
        self.gpu_id = gpu_id # return init_hidden ...
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=batch_first)

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # print ('init BasicRNNBlock')
        if self.gpu_id >= 0:
            return (autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda(self.gpu_id)), 
                autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda(self.gpu_id)))
        else:
            return (autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)))
    ### ??? !! batch_first: inputs.view(batch_size, -1, self.input_dim), self.hidden)
    ### ??? !! rnn_out return [:, -1, :]
    ## sequence length is not fixed, so I do not pass in sequence_length
    def forward(self, x):
        logger = logging.getLogger('gm_reg')
        logger.debug('Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('input size:')
        logger.debug (x.data.size())
        logger.debug ('input norm: %f', x.data.norm())
        if self.batch_first:
            batch_size = x.size()[0]
            lstm_out, self.hidden = self.lstm(
                x.view(batch_size, -1, self.input_dim), self.hidden)
        else:
            batch_size = x.size()[1]
            lstm_out, self.hidden = self.lstm(
                x.view(-1, batch_size, self.input_dim), self.hidden)
        logger.debug ('lstm_out size: ')
        logger.debug (lstm_out.data.size())
        logger.debug ('lstm_out norm: %f', lstm_out.data.norm())
        # print ('rnn_out norm: ', np.linalg.norm(rnn_out.data.cpu().numpy()))
        # print ('rnn_out/x ratio: ', np.linalg.norm(rnn_out.data.cpu().numpy())/np.linalg.norm(x.data.cpu().numpy()))
        return lstm_out

class BasicResRNNBlock(nn.Module):
    # no seq_length ?? batch_fist not true ??
    def __init__(self, gpu_id, input_dim, hidden_dim, batch_first):
        super(BasicResRNNBlock, self).__init__()
        self.gpu_id = gpu_id # return init_hidden ...
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, batch_first=batch_first)

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # print ('init BasicResRNNBlock')
        if self.gpu_id >= 0:
            return autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda(self.gpu_id))
        else:
            return autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))
    ### ??? !! batch_first: inputs.view(batch_size, -1, self.input_dim), self.hidden)
    ### ??? !! rnn_out return [:, -1, :]
    ## actually, this is useless of the input x is already organized according to batch_first
    def forward(self, x):
        residual = x
        # print ('residual norm: ', np.linalg.norm(residual.data.cpu().numpy()))
        logger = logging.getLogger('gm_reg')
        logger.debug('Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('input size:')
        logger.debug (x.data.size())
        logger.debug ('input norm: %f', x.data.norm())
        if self.batch_first:
            batch_size = x.size()[0]
            rnn_out, self.hidden = self.rnn(
                x.view(batch_size, -1, self.input_dim), self.hidden)
            rnn_out = rnn_out + residual.view(batch_size, -1, self.input_dim)
        else:
            batch_size = x.size()[1]
            rnn_out, self.hidden = self.rnn(
                x.view(-1, batch_size, self.input_dim), self.hidden)
            rnn_out = rnn_out + residual.view(-1, batch_size, self.input_dim)
        logger.debug ('rnn_out size: ')
        logger.debug (rnn_out.data.size())
        logger.debug ('rnn_out norm: %f', rnn_out.data.norm())
        # print ('rnn_out norm: ', np.linalg.norm(rnn_out.data.cpu().numpy()))
        # print ('rnn_out/residual ratio: ', np.linalg.norm(rnn_out.data.cpu().numpy())/np.linalg.norm(residual.data.cpu().numpy()))
        return rnn_out

class BasicResLSTMBlock(nn.Module):
    # no seq_length ?? batch_fist not true ??
    def __init__(self, gpu_id, input_dim, hidden_dim, batch_first):
        super(BasicResLSTMBlock, self).__init__()
        self.gpu_id = gpu_id # return init_hidden ...
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=batch_first)

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # print ('init BasicResRNNBlock')
        if self.gpu_id >= 0:
            return (autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda(self.gpu_id)),
                    autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda(self.gpu_id)))
        else:
            return (autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)),
                    autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)))
    ### ??? !! batch_first: inputs.view(batch_size, -1, self.input_dim), self.hidden)
    ### ??? !! rnn_out return [:, -1, :]
    ## actually, this is useless of the input x is already organized according to batch_first
    def forward(self, x):
        residual = x
        # print ('residual norm: ', np.linalg.norm(residual.data.cpu().numpy()))
        logger = logging.getLogger('gm_reg')
        logger.debug('Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('input size:')
        logger.debug (x.data.size())
        logger.debug ('input norm: %f', x.data.norm())
        if self.batch_first:
            batch_size = x.size()[0]
            lstm_out, self.hidden = self.lstm(
                x.view(batch_size, -1, self.input_dim), self.hidden)
            lstm_out = lstm_out + residual.view(batch_size, -1, self.input_dim)
        else:
            batch_size = x.size()[1]
            lstm_out, self.hidden = self.lstm(
                x.view(-1, batch_size, self.input_dim), self.hidden)
            lstm_out = lstm_out + residual.view(-1, batch_size, self.input_dim)
        logger.debug ('lstm_out size: ')
        logger.debug (lstm_out.data.size())
        logger.debug ('lstm_out norm: %f', lstm_out.data.norm())
        # print ('rnn_out norm: ', np.linalg.norm(rnn_out.data.cpu().numpy()))
        # print ('rnn_out/residual ratio: ', np.linalg.norm(rnn_out.data.cpu().numpy())/np.linalg.norm(residual.data.cpu().numpy()))
        return lstm_out

class ResNetRNN(nn.Module):
    # no seq_length ?? batch_fist not true ??
    def __init__(self, gpu_id, block, input_dim, hidden_dim, output_dim, blocks, batch_first):
        super(ResNetRNN, self).__init__()
        self.gpu_id = gpu_id
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_first = batch_first
        #for this one, I can use init_hidden to initialize hidden layer, also only return rnn_out
        self.rnn1 = BasicRNNBlock(gpu_id=gpu_id, input_dim=input_dim, hidden_dim=hidden_dim, batch_first=batch_first)
        self.layer1 = self._make_layer(gpu_id, block, hidden_dim, hidden_dim, blocks, batch_first)
        self.fc1 = InitLinear(hidden_dim, output_dim)

        logger = logging.getLogger('gm_reg')
        # ??? do I need this?
        for idx, m in enumerate(self.modules()):
            # print ('idx and self.modules():')
            # print (idx)
            # print (m)
            logger.info ('idx and self.modules():')
            logger.info (idx)
            logger.info (m)
            if isinstance(m, nn.Conv2d):
                logger.info ('initialization using kaiming_normal_')
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        for idx, m in enumerate(self.modules()):
            # print ('init hidden idx and self.modules():')
            # print ('init hidden idx: ', idx)
            # print ('init hidden m: ', m)
            if isinstance(m, BasicRNNBlock) or isinstance(m, BasicResRNNBlock):
                # print ('isinstance(m, BasicRNNBlock) or isinstance(m, BasicResRNNBlock)')
                m.hidden = m.init_hidden(batch_size)

    def _make_layer(self, gpu_id, block, input_dim, hidden_dim, blocks, batch_first):
        logger = logging.getLogger('gm_reg')
        layers = []
        layers.append(block(gpu_id, input_dim, hidden_dim, batch_first))
        for i in range(1, blocks):
            layers.append(block(gpu_id, input_dim, hidden_dim, batch_first)) # ?? need init_hidden()??
        for layer in layers:
            layer.register_forward_hook(get_features_hook)
        logger.info ('layers: ')
        logger.info (layers)
        return nn.Sequential(*layers)

    def forward(self, x):
        logger = logging.getLogger('gm_reg')
        logger.debug('x shape')
        logger.debug (x.shape)
        if self.batch_first:
            batch_size = x.size()[0]
        else:
            batch_size = x.size()[1]
        x = self.rnn1(x)
        features.append(x.data)
        logger.debug('Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('before blocks size:')
        logger.debug (x.data.size())
        logger.debug ('before blocks norm: %f', x.data.norm())
        x = self.layer1(x)
        logger.debug ('after blocks size:')
        logger.debug (x.data.size())
        logger.debug ('after blocks norm: %f', x.data.norm())
        ## ?? softmax loss
        if self.batch_first:
            x = F.sigmoid(self.fc1(x[:, -1, :].view(batch_size, -1)))
        else:
            x = F.sigmoid(self.fc1(x[-1, :, :].view(batch_size, -1)))
        return x

class ResNetLSTM(nn.Module):
    # no seq_length ?? batch_fist not true ??
    def __init__(self, gpu_id, block, input_dim, hidden_dim, output_dim, blocks, batch_first):
        super(ResNetLSTM, self).__init__()
        self.gpu_id = gpu_id
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_first = batch_first
        #for this one, I can use init_hidden to initialize hidden layer, also only return rnn_out
        self.lstm1 = BasicLSTMBlock(gpu_id=gpu_id, input_dim=input_dim, hidden_dim=hidden_dim, batch_first=batch_first)
        self.layer1 = self._make_layer(gpu_id, block, hidden_dim, hidden_dim, blocks, batch_first)
        self.fc1 = InitLinear(hidden_dim, output_dim)

        logger = logging.getLogger('gm_reg')
        # ??? do I need this?
        for idx, m in enumerate(self.modules()):
            print ('idx and self.modules():')
            print (idx)
            print (m)
            logger.info ('idx and self.modules():')
            logger.info (idx)
            logger.info (m)
            if isinstance(m, nn.Conv2d):
                logger.info ('initialization using kaiming_normal_')
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        for idx, m in enumerate(self.modules()):
            logger.debug ('init hidden idx and self.modules():')
            # print ('init hidden idx: ', idx)
            logger.debug ('init hidden m: ')
            logger.debug (m)
            if isinstance(m, BasicLSTMBlock) or isinstance(m, BasicResLSTMBlock):
                logger.debug ('isinstance(m, BasicLSTMBlock) or isinstance(m, BasicResLSTMBlock)')
                m.hidden = m.init_hidden(batch_size)

    def _make_layer(self, gpu_id, block, input_dim, hidden_dim, blocks, batch_first):
        logger = logging.getLogger('gm_reg')
        layers = []
        layers.append(block(gpu_id, input_dim, hidden_dim, batch_first))
        for i in range(1, blocks):
            layers.append(block(gpu_id, input_dim, hidden_dim, batch_first)) # ?? need init_hidden()??
        for layer in layers:
            layer.register_forward_hook(get_features_hook)
        logger.info ('layers: ')
        logger.info (layers)
        print ('layers: ')
        return nn.Sequential(*layers)

    def forward(self, x):
        logger = logging.getLogger('gm_reg')
        logger.debug('x shape')
        logger.debug (x.shape)
        if self.batch_first:
            batch_size = x.size()[0]
        else:
            batch_size = x.size()[1]
        x = self.lstm1(x)
        features.append(x.data)
        logger.debug('Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('before blocks size:')
        logger.debug (x.data.size())
        logger.debug ('before blocks norm: %f', x.data.norm())
        x = self.layer1(x)
        logger.debug ('after blocks size:')
        logger.debug (x.data.size())
        logger.debug ('after blocks norm: %f', x.data.norm())
        ## ?? softmax loss
        if self.batch_first:
            x = F.sigmoid(self.fc1(x[:, -1, :].view(batch_size, -1)))
        else:
            x = F.sigmoid(self.fc1(x[-1, :, :].view(batch_size, -1)))
        return x

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class WLMResNetLSTM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, gpu_id, block, ntoken, input_dim, hidden_dim, blocks, batch_first, tie_weights=False):
        super(WLMResNetLSTM, self).__init__()
        self.gpu_id = gpu_id
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.encoder = nn.Embedding(ntoken, input_dim)
        self.lstm1 = BasicLSTMBlock(gpu_id=gpu_id, input_dim=input_dim, hidden_dim=hidden_dim, batch_first=batch_first)
        self.layer1 = self._make_layer(gpu_id, block, hidden_dim, hidden_dim, blocks, batch_first)
        self.decoder = nn.Linear(hidden_dim, ntoken)
        
        logger = logging.getLogger('gm_reg')
        # ??? do I need this?
        for idx, m in enumerate(self.modules()):
            print ('idx and self.modules():')
            print (idx)
            print (m)
            logger.info ('idx and self.modules():')
            logger.info (idx)
            logger.info (m)
            if isinstance(m, nn.Conv2d):
                logger.info ('initialization using kaiming_normal_')
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if hidden_dim != input_dim:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        self.init_weights()


    
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        for idx, m in enumerate(self.modules()):
            logger.debug ('init hidden idx and self.modules():')
            # print ('init hidden idx: ', idx)
            logger.debug ('init hidden m: ')
            logger.debug (m)
            if isinstance(m, BasicLSTMBlock) or isinstance(m, BasicResLSTMBlock):
                logger.debug ('isinstance(m, BasicLSTMBlock) or isinstance(m, BasicResLSTMBlock)')
                m.hidden = m.init_hidden(batch_size)

    def repackage_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        for idx, m in enumerate(self.modules()):
            logger.debug ('init hidden idx and self.modules():')
            # print ('init hidden idx: ', idx)
            logger.debug ('init hidden m: ')
            logger.debug (m)
            if isinstance(m, BasicLSTMBlock) or isinstance(m, BasicResLSTMBlock):
                logger.debug ('wlm repackage_hidden isinstance(m, BasicLSTMBlock) or isinstance(m, BasicResLSTMBlock)')
                # print ('before repackage m.hidden: ', m.hidden)
                m.hidden = repackage_hidden(m.hidden)
                # print ('after repackage m.hidden: ', m.hidden)
    
    def _make_layer(self, gpu_id, block, input_dim, hidden_dim, blocks, batch_first):
        logger = logging.getLogger('gm_reg')
        layers = []
        layers.append(block(gpu_id, input_dim, hidden_dim, batch_first))
        for i in range(1, blocks):
            layers.append(block(gpu_id, input_dim, hidden_dim, batch_first)) # ?? need init_hidden()??
        for layer in layers:
            layer.register_forward_hook(get_features_hook)
        logger.info ('layers: ')
        logger.info (layers)
        print ('layers: ')
        return nn.Sequential(*layers)


    def forward(self, x):
        logger = logging.getLogger('gm_reg')
        logger.debug('x shape')
        logger.debug (x.shape)
        if self.batch_first:
            batch_size = x.size()[0]
        else:
            batch_size = x.size()[1]
        x = self.encoder(x)
        logger.debug('after encoder x shape')
        logger.debug (x.shape)
        x = self.lstm1(x)
        features.append(x.data)
        logger.debug('Inside ' + self.__class__.__name__ + ' forward')
        logger.debug ('before blocks size:')
        logger.debug (x.data.size())
        logger.debug ('before blocks norm: %f', x.data.norm())
        x = self.layer1(x)
        logger.debug ('after blocks size:')
        logger.debug (x.data.size())
        logger.debug ('after blocks norm: %f', x.data.norm())
        decoded = self.decoder(x.view(x.size(0)*x.size(1), x.size(2)))
        return decoded.view(x.size(0), x.size(1), decoded.size(1))


def get_features_hook(module, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    
    logger = logging.getLogger('gm_reg')
    logger.debug('Inside ' + module.__class__.__name__ + ' forward hook')
    logger.debug('')
    # logger.debug('input:')
    # logger.debug(input)
    logger.debug('input: ')
    logger.debug(type(input))
    logger.debug('input[0]: ')
    logger.debug(type(input[0]))
    logger.debug('output: ')
    logger.debug(type(output))
    logger.debug('')
    logger.debug('input[0] size:')
    logger.debug(input[0].size())
    logger.debug('input norm: %f', input[0].data.norm())
    logger.debug('output size:')
    logger.debug(output.data.size())
    logger.debug('output norm: %f', output.data.norm())
    features.append(output.data)

def get_batch(source, i, seqnum):
    seq_len = min(seqnum, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def batchify(data, bsz, gpu_id):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    print ("data shape: ", data.shape)
    return data.cuda(gpu_id)

def chunk_array(arr, chunks, dim):
    print ("chunk_array original shape: ", arr.shape)
    if dim == 0:
        chunk_array_list = []
        base = int(arr.shape[0] / chunks)
        for i in range(chunks):
            chunk_array_list.append(arr[i * base: (i+1) * base])
    return chunk_array_list

def train(rnn, gpu_id, train_loader, test_loader, criterion, optimizer, momentum_mu, blocks, n_hidden, weightdecay, firstepochs, labelnum, batch_first, n_epochs, model_name, hyperpara_list, gm_num, gm_lambda_ratio_value, uptfreq):
    logger = logging.getLogger('gm_reg')
    # res_regularizer_instance = ResRegularizer(prior_beta=prior_beta, reg_lambda=reg_lambda, momentum_mu=momentum_mu, blocks=blocks, feature_dim=n_hidden, model_name=model_name)
    # Keep track of losses for plotting
    start = time.time()
    st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
    print(st)

    opt = gm_prior_optimizer_pytorch.GMOptimizer()
    for name, f in rnn.named_parameters():
        if "weight_ih" in name or "weight_hh" in name:
            # print ("lstm weight, needed to be divided into four gates")
            w_array_chunk = chunk_array(f.data.cpu().numpy(),4,0)
            opt.gm_register(name+"_first_gate", w_array_chunk[0], model_name, hyperpara_list, gm_num, gm_lambda_ratio_value, uptfreq)
            opt.gm_register(name+"_second_gate", w_array_chunk[1], model_name, hyperpara_list, gm_num, gm_lambda_ratio_value, uptfreq)
            opt.gm_register(name+"_third_gate", w_array_chunk[2], model_name, hyperpara_list, gm_num, gm_lambda_ratio_value, uptfreq)
            opt.gm_register(name+"_fourth_gate", w_array_chunk[3], model_name, hyperpara_list, gm_num, gm_lambda_ratio_value, uptfreq)
        else:
            opt.gm_register(name, f.data.cpu().numpy(), model_name, hyperpara_list, gm_num, gm_lambda_ratio_value, uptfreq)
    opt.weightdimSum = sum(opt.weight_dim_list.values())
    print ("opt.weightdimSum: ", opt.weightdimSum)
    print ("opt.weight_name_list: ", opt.weight_name_list)
    print ("opt.weight_dim_list: ", opt.weight_dim_list)
    print ("opt.gmregularizers: ", opt.gmregularizers)
    print ("len(opt.gmregularizers): ", len(opt.gmregularizers))
    print ("len(opt.weight_name_list): ", len(opt.weight_name_list))

    pre_running_loss = 0.0
    for epoch in range(n_epochs):
        running_loss = 0.0
        running_accuracy = 0.0
        # output, loss = train(model_name, epoch, batch_size, batch_first, category_tensor, line_tensor, res_regularizer_instance)
        for batch_idx, data_iter in enumerate(train_loader, 0):
            data_x, data_y = data_iter
            # print ('data_y shape: ', data_y.shape)
            data_x, data_y = Variable(data_x.cuda(gpu_id)), Variable(data_y.cuda(gpu_id))
            rnn.init_hidden(data_x.shape[0])
            optimizer.zero_grad()
            features.clear()
            logger.debug ('data_x: ')
            logger.debug (data_x)
            logger.debug ('data_x norm: %f', data_x.norm())
            outputs = rnn(data_x)
            loss = criterion(outputs, data_y)
            logger.debug ("features length: %d", len(features))
            for feature in features:
                logger.debug ("feature size:")
                logger.debug (feature.data.size())
                logger.debug ("feature norm: %f", feature.data.norm())
            accuracy = AUCAccuracy(outputs.data.cpu().numpy(), data_y.data.cpu().numpy())[0]
            loss.backward()
            ### print norm
            if (epoch == 0 and batch_idx < 1000) or batch_idx % 1000 == 0:
                for name, f in rnn.named_parameters():
                    print ('batch_idx: ', batch_idx)
                    print ('param name: ', name)
                    print ('param size:', f.data.size())
                    print ('param norm: ', np.linalg.norm(f.data.cpu().numpy()))
                    print ('lr 1.0 * param grad norm: ', np.linalg.norm(f.grad.data.cpu().numpy() * 1.0))
            ### when to use gm_reg
            # begin GM Reg
            if "reg" in model_name and epoch >= firstepochs:
                for name, f in rnn.named_parameters():
                    # print ("len(trainloader.dataset): ", len(trainloader.dataset))
                    if "weight_ih" in name or "weight_hh" in name:
                        # print ("lstm weight, needed to be divided into four gates")
                        # hard code here!!
                        opt.apply_GM_regularizer_constraint(labelnum, len(train_loader.dataset), epoch, weightdecay, f, name+"_first_gate", batch_idx)
                        opt.apply_GM_regularizer_constraint(labelnum, len(train_loader.dataset), epoch, weightdecay, f, name+"_second_gate", batch_idx)
                        opt.apply_GM_regularizer_constraint(labelnum, len(train_loader.dataset), epoch, weightdecay, f, name+"_third_gate", batch_idx)
                        opt.apply_GM_regularizer_constraint(labelnum, len(train_loader.dataset), epoch, weightdecay, f, name+"_fourth_gate", batch_idx)
                    else:
                        opt.apply_GM_regularizer_constraint(labelnum, len(train_loader.dataset), epoch, weightdecay, f, name, batch_idx)
            # end GM Reg
            ### print norm
            optimizer.step()
            running_loss += loss.item() * len(data_x)
            # print ('check!! len(data_x) --> last batch? : ', len(data_x))
            running_accuracy += accuracy * len(data_x)

        # Print epoch number, loss, name and guess
        # print ('maximum batch_idx: ', batch_idx)
        running_loss = running_loss / len(train_loader.dataset)
        running_accuracy = running_accuracy / len(train_loader.dataset)
        print('epoch: %d, training loss per sample per label =  %f, training accuracy =  %f'%(epoch, running_loss, running_accuracy))
        print('abs(running_loss - pre_running_loss)', abs(running_loss - pre_running_loss))
        pre_running_loss = running_loss

        # test
        outputs_list = []
        labels_list = []
        test_loss = 0.0
        with torch.no_grad():
            for batch_idx, data_iter in enumerate(test_loader):
                data_x, data_y = data_iter
                # print ("test data_y shape: ", data_y.shape)
                data_x, data_y = Variable(data_x.cuda(gpu_id)), Variable(data_y.cuda(gpu_id))
                rnn.init_hidden(data_x.shape[0])
                outputs = rnn(data_x)
                # print ('outputs shape: ', outputs.shape)
                # print ('data_y shape: ', data_y.shape)
                loss = criterion(outputs, data_y)
                test_loss += loss.item() * len(data_x)
                outputs_list.extend(list(outputs.data.cpu().numpy()))
                labels_list.extend(list(data_y.data.cpu().numpy()))
            # print ('test outputs_list length: ', len(outputs_list))
            # print ('test labels_list length: ', len(labels_list))
            metrics = AUCAccuracy(np.array(outputs_list), np.array(labels_list))
            accuracy, macro_auc, micro_auc = metrics[0], metrics[1], metrics[2]
            # print ('maximum test batch_idx: ', batch_idx)
            test_loss = test_loss / len(test_loader.dataset)
            print ('test loss = %f, test accuracy = %f, test macro auc = %f, test micro auc = %f'%(test_loss, accuracy, macro_auc, micro_auc))
            if epoch == (n_epochs - 1):
                print ('| final a {:.10f} | final b {:.10f} | final alpha {:.10f} | final gm_num {:d} | final gm_lambda_ratio {:.10f} | final  gmuptfreq {:d} | final paramuptfreq {:d} | final weight_decay {:.10f}'.format(hyperpara_list[0], hyperpara_list[1], hyperpara_list[2], gm_num, gm_lambda_ratio_value, uptfreq[0], uptfreq[1], weightdecay))
                print ('final test loss = %f, test accuracy = %f, test macro auc = %f, test micro auc = %f'%(test_loss, accuracy, macro_auc, micro_auc))

    done = time.time()
    do = datetime.datetime.fromtimestamp(done).strftime('%Y-%m-%d %H:%M:%S')
    print (do)
    elapsed = done - start
    print (elapsed)
    print('Finished Training')


def adjust_learning_rate(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
        print ("new param_group['lr']: ", param_group['lr'])

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def trainwlm(model_name, rnn, gpu_id, corpus, batchsize, train_data, val_data, test_data, seqnum, clip, criterion, optimizer, momentum_mu, blocks, n_hidden, weightdecay, firstepochs, labelnum, batch_first, n_epochs):
    logger = logging.getLogger('gm_reg')
    # res_regularizer_instance = ResRegularizer(prior_beta=prior_beta, reg_lambda=reg_lambda, momentum_mu=momentum_mu, blocks=blocks, feature_dim=n_hidden, model_name=model_name)
    # Keep track of losses for plotting
    start = time.time()
    st = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
    print(st)
    pre_running_loss = 0.0
    best_val_loss = None
    for epoch in range(n_epochs):
        rnn.train()
        total_loss = 0.
        ntokens = len(corpus.dictionary)
        rnn.init_hidden(batchsize) # since the data is batchfied, so batchsize can be just passed in from main()
        for batch_idx, i in enumerate(range(0, train_data.size(0) - 1, seqnum)):
            data_x, data_y = get_batch(train_data, i, seqnum)
            # print ('data_y shape: ', data_y.shape)
            data_x, data_y = Variable(data_x.cuda(gpu_id)), Variable(data_y.cuda(gpu_id))

            rnn.repackage_hidden()
            optimizer.zero_grad()
            features.clear()
            logger.debug ('data_x: ')
            logger.debug (data_x)
            logger.debug ('data_x shape: ')
            logger.debug (data_x.shape)
            # logger.debug ('data_x norm: %f', data_x.norm())
            outputs = rnn(data_x)
            loss = criterion(outputs.view(-1, ntokens), data_y)
            logger.debug ("features length: %d", len(features))
            for feature in features:
                logger.debug ("feature size:")
                logger.debug (feature.data.size())
                logger.debug ("feature norm: %f", feature.data.norm())
            loss.backward()
            # print ("batch_idx: ", batch_idx)
            ### print norm
            if (epoch == 0 and batch_idx < 1000) or batch_idx % 1000 == 0:
                for name, f in rnn.named_parameters():
                    print ('batch_idx: ', batch_idx)
                    print ('param name: ', name)
                    print ('param size:', f.data.size())
                    print ('param norm: ', np.linalg.norm(f.data.cpu().numpy()))
                    print ('lr 1.0 * param grad norm: ', np.linalg.norm(f.grad.data.cpu().numpy() * 1.0))
            ### when to use gm_reg
                
            ### print norm
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), clip)
            optimizer.step()
            total_loss += loss.item()
            # print ('check!! len(data_x) --> last batch? : ', len(data_x))

        # Print epoch number, loss, name and guess
        # print ('maximum batch_idx: ', batch_idx)
        # actually the last mini-batch may contain less time-steps!! originally, the train loss is printed every args.log_interval mini-batches
        # not totally correct, but just show some hints, then ok
        cur_loss = total_loss / (batch_idx+1)
        print('| epoch {:3d} | lr {:.8f} | {:5d} batches '
                    'loss per sample per timestep {:.8f} | ppl {:8.2f}'.format(
                epoch, get_lr(optimizer), batch_idx, cur_loss, math.exp(cur_loss)))
        print ('abs(cur_loss - pre_running_loss)', abs(cur_loss - pre_running_loss))
        pre_running_loss = cur_loss
        total_loss = 0

        # validation
        # Turn on evaluation mode which disables dropout.
        rnn.eval()
        total_val_loss = 0.
        ntokens = len(corpus.dictionary)
        print ('ntokens: ', ntokens)
        rnn.init_hidden(batchsize)
        with torch.no_grad():
            for batch_idx in range(0, val_data.size(0) - 1, seqnum):
                data_x, data_y = get_batch(val_data, batch_idx, seqnum)
                # print ("val data_y shape: ", data_y.shape)
                data_x, data_y = Variable(data_x.cuda(gpu_id)), Variable(data_y.cuda(gpu_id))
                outputs = rnn(data_x)
                outputs_flat = outputs.view(-1, ntokens)
                # print ('outputs_flat shape: ', outputs_flat.shape)
                # print ('data_y shape: ', data_y.shape)
                # sum over timesteps, this is absolutely correct even if the last mini-batch is not equal lenght of timesteps
                total_val_loss += len(data_x) * criterion(outputs_flat, data_y).item()
                rnn.repackage_hidden()
        average_val_loss = total_val_loss / (len(val_data) - 1)
        print('=' * 89)
        print('| End of training | val loss {:.8f} | val ppl {:8.2f}'.format(average_val_loss, math.exp(average_val_loss)))
        print('=' * 89)
        if not best_val_loss or average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
        else:
            adjust_learning_rate(optimizer, float(1/4.0))
        
        
        # test
        # Turn on evaluation mode which disables dropout.
        rnn.eval()
        total_test_loss = 0.
        ntokens = len(corpus.dictionary)
        print ('ntokens: ', ntokens)
        rnn.init_hidden(batchsize)
        with torch.no_grad():
            for batch_idx in range(0, test_data.size(0) - 1, seqnum):
                data_x, data_y = get_batch(test_data, batch_idx, seqnum)
                # print ("test data_y shape: ", data_y.shape)
                data_x, data_y = Variable(data_x.cuda(gpu_id)), Variable(data_y.cuda(gpu_id))
                outputs = rnn(data_x)
                outputs_flat = outputs.view(-1, ntokens)
                # print ('outputs_flat shape: ', outputs_flat.shape)
                # print ('data_y shape: ', data_y.shape)
                # sum over timesteps, this is absolutely correct even if the last mini-batch is not equal lenght of timesteps
                total_test_loss += len(data_x) * criterion(outputs_flat, data_y).item()
                rnn.repackage_hidden()
        average_test_loss = total_test_loss / (len(test_data) - 1)
        print('=' * 89)
        print('| End of training | test loss {:.8f} | test ppl {:8.2f}'.format(average_test_loss, math.exp(average_test_loss)))
        print('=' * 89)
        if epoch == (n_epochs - 1):
            print('=' * 89)
            print('| final weightdecay {:.10f} | final prior_beta {:.10f} | final reg_lambda {:.10f}'.format(weightdecay, prior_beta, reg_lambda))
            print('| End of training | final test loss {:.8f} | final test ppl {:8.2f}'.format(average_test_loss, math.exp(average_test_loss)))
            print('=' * 89)
            

    done = time.time()
    do = datetime.datetime.fromtimestamp(done).strftime('%Y-%m-%d %H:%M:%S')
    print (do)
    elapsed = done - start
    print (elapsed)
    print('Finished Training')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Residual LSTM')
    parser.add_argument('-traindatadir', type=str, help='training data directory, also the data dir for word language model')
    parser.add_argument('-trainlabeldir', type=str, help='training label directory')
    parser.add_argument('-testdatadir', type=str, help='test data directory')
    parser.add_argument('-testlabeldir', type=str, help='test label directory')
    parser.add_argument('-seqnum', type=int, help='sequence number')
    parser.add_argument('-modelname', type=str, help='resnetrnn or reslstm or rnn or lstm')
    parser.add_argument('-blocks', type=int, help='number of blocks')
    parser.add_argument('-lr', type=float, help='0.001 for MIMIC-III')
    parser.add_argument('-weightdecay', type=float, help='weightdecay')
    parser.add_argument('-batchsize', type=int, help='batch_size')
    parser.add_argument('-firstepochs', type=int, help='first epochs when no regularization is imposed')
    parser.add_argument('-considerlabelnum', type=int, help='just a reminder, need to consider label number because the loss is averaged across labels')
    parser.add_argument('-maxepoch', type=int, help='max_epoch')
    parser.add_argument('-gmnum', type=int, help='gm_number')
    parser.add_argument('-gmuptfreq', type=int, help='gm update frequency, in steps')
    parser.add_argument('-paramuptfreq', type=int, help='parameter update frequency, in steps')
    # parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('-gpuid', type=int, help='gpuid')
    parser.add_argument('--batch_first', action='store_true')
    parser.add_argument('--emsize', type=int, default=200, help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200, help='number of hidden units per layer, 200 for word language model, 128 for other datasets')
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument('--tied', action='store_true', help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    # torch.manual_seed(args.seed)
    print ("args.debug: ", args.debug)
    print ("wlm args.batch_first: ", args.batch_first)
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, filename="./logfile", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, filename="./logfile", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
    logger = logging.getLogger('gm_reg')
    logger.info ('#################################')
    gpu_id = args.gpuid
    print('gpu_id: ', gpu_id)

    ######################################################################
    # Load Data
    # ---------
    #
    if "wikitext" not in args.traindatadir:
        print ("not word language model")
        print ("loading train_x")
        # train_x = np.genfromtxt(args.traindatadir, dtype=np.float32, delimiter=',')
        if "MIMIC-III" in args.traindatadir: 
            train_x_sparse_matrix = scipy.sparse.load_npz(args.traindatadir)
            train_x_sparse_matrix = train_x_sparse_matrix.astype(np.float32)
            train_x = np.array(train_x_sparse_matrix.todense())
        else:
            train_x = np.genfromtxt(args.traindatadir, dtype=np.float32, delimiter=',')
        
        train_y = np.genfromtxt(args.trainlabeldir, dtype=np.float32, delimiter=',')
        train_y = train_y.reshape((train_y.shape[0],-1))
        print ("loading test_x")
        # test_x = np.genfromtxt(args.testdatadir, dtype=np.float32, delimiter=',')
        if "MIMIC-III" in args.traindatadir:
            test_x_sparse_matrix = scipy.sparse.load_npz(args.testdatadir)
            test_x_sparse_matrix = test_x_sparse_matrix.astype(np.float32)
            test_x = np.array(test_x_sparse_matrix.todense())
        else:
            test_x = np.genfromtxt(args.testdatadir, dtype=np.float32, delimiter=',')
        
        test_y = np.genfromtxt(args.testlabeldir, dtype=np.float32, delimiter=',')
        test_y = test_y.reshape((test_y.shape[0],-1))
        train_x = train_x.reshape((train_x.shape[0], args.seqnum, -1))
        test_x = test_x.reshape((test_x.shape[0], args.seqnum, -1))
        print ('train_x.shape: ', train_x.shape)
        print ('test_x.shape: ', test_x.shape)
        print ('train_y.shape: ', train_y.shape)
        print ('test_y.shape: ', test_y.shape)
        input_dim = train_x.shape[-1]
        print ('check input_dim: ', input_dim)

        train_dataset = Data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
        train_loader = Data.DataLoader(dataset=train_dataset,
                                       batch_size=args.batchsize,
                                       shuffle=True)
        print ('check len(train_dataset): ', len(train_dataset))
        test_dataset = Data.TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
        test_loader = Data.DataLoader(dataset=test_dataset,
                                       batch_size=args.batchsize,
                                       shuffle=True)
        print ('check len(test_dataset): ', len(test_dataset))

        label_num = train_y.shape[1]
        print ("check label number: ", label_num)
    else:
        label_num = 1
        print ("wlm check label number (hard code??): ", label_num)
        corpus = data.Corpus(args.traindatadir)
        train_data = batchify(corpus.train, args.batchsize, gpu_id)
        val_data = batchify(corpus.valid, args.batchsize, gpu_id)
        test_data = batchify(corpus.test, args.batchsize, gpu_id)
        ntokens = len(corpus.dictionary)
        input_dim = args.emsize

    ########## using for
    gm_lambda_ratio_list = [ -1.]
    a_list = [1e-1, 1e-2]
    b_list, alpha_list = [100., 50., 20., 10., 5., 2., 1., 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001], [0.7, 0.5, 0.3, 0.9]

    for alpha_idx in range(len(alpha_list)):
        for b_idx in range(len(b_list)):
            for a_idx in range(len(a_list)):
                alpha_value = alpha_list[alpha_idx]
                b_value = b_list[b_idx]
                a_value = a_list[a_idx]
                gm_lambda_ratio_value = random.choice(gm_lambda_ratio_list)

                ########## using for
                n_hidden = args.nhid
                n_epochs = args.maxepoch

                if "wikitext" not in args.traindatadir:
                    if "res" in args.modelname and "rnn" in args.modelname:
                        print ('check resrnn model')
                        rnn = ResNetRNN(args.gpuid, BasicResRNNBlock, input_dim, n_hidden, label_num, args.blocks, args.batch_first)
                        rnn = rnn.cuda(args.gpuid)
                    elif "res" in args.modelname and "lstm" in args.modelname:
                        print ('check reslstm model')
                        rnn = ResNetLSTM(args.gpuid, BasicResLSTMBlock, input_dim, n_hidden, label_num, args.blocks, args.batch_first)
                        rnn = rnn.cuda(args.gpuid)
                    elif "res" not in args.modelname and "rnn" in args.modelname:
                        print ('check rnn model')
                        rnn = ResNetRNN(args.gpuid, BasicRNNBlock, input_dim, n_hidden, label_num, args.blocks, args.batch_first)
                        rnn = rnn.cuda(args.gpuid)
                    elif "res" not in args.modelname and "lstm" in args.modelname:
                        print ('check lstm model')
                        rnn = ResNetLSTM(args.gpuid, BasicLSTMBlock, input_dim, n_hidden, label_num, args.blocks, args.batch_first)
                        rnn = rnn.cuda(args.gpuid)
                    else:
                        print("Invalid model name, exiting...")
                        exit()
                else:
                    if "res" in args.modelname and "lstm" in args.modelname:
                        rnn = WLMResNetLSTM(args.gpuid, BasicResLSTMBlock, ntokens, input_dim, n_hidden, args.blocks, args.batch_first, tie_weights=args.tied)
                        rnn = rnn.cuda(args.gpuid)
                    elif "res" not in args.modelname and "lstm" in args.modelname:
                        rnn = WLMResNetLSTM(args.gpuid, BasicLSTMBlock, ntokens, input_dim, n_hidden, args.blocks, args.batch_first, tie_weights=args.tied)
                        rnn = rnn.cuda(args.gpuid)
                    else:
                        print("Invalid model name, exiting...")
                        exit()


                if "reg" in args.modelname:
                    print ('optimizer without wd')
                    # optimizer = Adam(rnn.parameters(), lr=args.lr)
                    if "wikitext" not in args.traindatadir:
                        optimizer = optim.SGD(rnn.parameters(), lr=args.lr, momentum=0.9)
                    else:
                        optimizer = optim.SGD(rnn.parameters(), lr=args.lr)
                else:
                    print ('optimizer with wd')
                    # optimizer = Adam(rnn.parameters(), lr=args.lr, weight_decay=args.decay)
                    if "wikitext" not in args.traindatadir:
                        optimizer = optim.SGD(rnn.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weightdecay)
                    else:
                        optimizer = optim.SGD(rnn.parameters(), lr=args.lr, weight_decay=args.weightdecay)
                    # optimizer = optim.SGD(rnn.parameters(), lr=args.lr, weight_decay=args.decay)
                print ('optimizer: ', optimizer)

                if "wikitext" not in args.traindatadir:
                    criterion = nn.BCELoss()
                else:
                    criterion = nn.CrossEntropyLoss()
                print ('criterion: ', criterion)
                momentum_mu = 0.9 # momentum mu
                if "wikitext" not in args.traindatadir:
                    train(rnn, args.gpuid, train_loader, test_loader, criterion, optimizer, momentum_mu, args.blocks, n_hidden, args.weightdecay, args.firstepochs, label_num, args.batch_first, args.maxepoch, args.modelname, [a_value, b_value, alpha_value], args.gmnum, gm_lambda_ratio_value, [args.gmuptfreq, args.paramuptfreq])
                else:
                    trainwlm(args.modelname, rnn, args.gpuid, corpus, args.batchsize, train_data, val_data, test_data, args.seqnum, args.clip, criterion, optimizer, momentum_mu, args.blocks, n_hidden, args.weightdecay, args.firstepochs, label_num, args.batch_first, args.maxepoch)

####### real and real_wlm
# CUDA_VISIBLE_DEVICES=0 python train_lstm_main_hook_gmreg_real_wlm_tune_hyperparam.py -traindatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_x_seq_sparse.npz -trainlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_train_y_seq.csv -testdatadir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_x_seq_sparse.npz -testlabeldir /hdd1/zhaojing/res-regularization/MIMIC-III-dataset/formal_test_y_seq.csv -seqnum 9 -modelname reglstm -blocks 1 -lr 1.0 -weightdecay 0.001 -batchsize 100 -firstepochs 0 -considerlabelnum 1 -maxepoch 500 -gmnum 4 -gmuptfreq 100 -paramuptfreq 50 -gpuid 0 --batch_first --nhid 128
