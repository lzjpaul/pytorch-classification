# This file contains all the classes that are related to 
# GM-prior adaptive regularizer.
# =============================================================================
import torch
import numpy as np
from scipy.stats import norm as gaussian
import math

class GMOptimizer():
    '''
    introduce hyper-parameters for GM-regularization: a, b, alpha
    '''
    def __init__(self):
        self.weight_name_list = {}
        self.weight_dim_list = {}
        self.gmregularizers = {}

    def layer_wise_hyperpara(self, fea_num, hyperpara_list):
        print ("layer_wise fea_num: ", fea_num)
        alpha_val = fea_num ** (hyperpara_list[2])
        b_val = hyperpara_list[1] * fea_num
        a_val = 1. + b_val * hyperpara_list[0]
        return [a_val, b_val, alpha_val]

    # value is numpy here
    def gm_register(self, name, value, model_name, hyperpara_list, gm_num, gm_lambda_ratio_value, uptfreq):
        print ("param name: ", name)
        print ("param shape: ", value.shape)
        if np.ndim(value) == 2:
            self.weight_name_list[name] = name
            dims = value.shape[0] * value.shape[1]
            print ("dims: ", dims)
            self.weight_dim_list[name] = dims
            layer_hyperpara = self.layer_wise_hyperpara(dims, hyperpara_list)
            pi = [1.0/gm_num for _ in range(gm_num)]
            k = 1.0 + gm_lambda_ratio_value
            print ("gm_lambda_ratio_value: ", gm_lambda_ratio_value)
            # calculate base
            if model_name == 'alexnet':
                if 'conv1' in name:
                    base = 100000000.0 / 10000000.0
                else:
                    base = 10000.0 / 1000.0
            else: # for resnet
                if 'conv' in name:
                    base = (3.0 * 3.0 * value.shape[0] / 2.0) / 10.0
                else:
                    base = ((value.shape[0] + value.shape[1]) / 6.0) / 10.0
            print ("base: ", base)
            # calculate GM initialized lambda (1/variance)
            if gm_lambda_ratio_value >= 0.0:
                reg_lambda = [base*math.pow(k,_) for _ in  range(gm_num)]
            else:
                reg_lambda_range = base * float(gm_num)
                reg_lambda = np.arange(1.0, reg_lambda_range, reg_lambda_range/gm_num)
            self.gmregularizers[name] = GMRegularizer(hyperpara=layer_hyperpara, gm_num=gm_num, pi=pi, reg_lambda=reg_lambda, uptfreq=uptfreq)

    def apply_GM_regularizer_constraint(self, trainnum, epoch, weight_decay, f, name, step):
        # if np.ndim(tensor.to_numpy(value)) <= 2:
        if np.ndim(f.data.cpu().numpy()) != 2:
            f.grad.data.add_(float(weight_decay), f.data)
        else: # weight parameter
            self.gmregularizers[name].apply(trainnum, epoch, f, name, step)


class GMRegularizer():
    '''GM regularization
    Args:
        hyperparameters: a, b, alpha (like the coefficient of L2), uptfreq
    '''

    def __init__(self, hyperpara=None, gm_num=None, pi=None, reg_lambda=None, uptfreq=None):
        self.a, self.b, self.alpha, self.gm_num = hyperpara[0], hyperpara[1], hyperpara[2], gm_num
        print ("init self.a, self.b, self.alpha, self.gm_num: ", self.a, self.b, self.alpha, self.gm_num)
        self.pi, self.reg_lambda = np.reshape(np.array(pi), (1, gm_num)), np.reshape(np.array(reg_lambda), (1, gm_num))
        print ("init self.reg_lambda: ", self.reg_lambda)
        print ("init self.pi: ", self.pi)
        self.gmuptfreq, self.paramuptfreq = uptfreq[0], uptfreq[1]
        print ("init self.gmuptfreq, self.paramuptfreq: ", self.gmuptfreq, self.paramuptfreq)

    # calc the resposibilities for pj(wi)
    def calcResponsibility(self):
        # responsibility normalized with pi
        responsibility = gaussian.pdf(self.w_array, loc=np.zeros(shape=(1, self.gm_num)), scale=1/np.sqrt(self.reg_lambda))*self.pi
        # responsibility normalized with summation(denominator)
        self.responsibility = responsibility/(np.sum(responsibility, axis=1).reshape(self.w_array.shape))
    
    def update_GM_Prior_EM(self, name, step):
        # update pi
        self.reg_lambda = (2 * (self.a - 1) + np.sum(self.responsibility, axis=0)) / (2 * self.b + np.sum(self.responsibility * np.square(self.w_array), axis=0))
        if step % self.gmuptfreq == 0:
            print ("name: ", name)
            print ("np.sum(self.responsibility, axis=0): ", np.sum(self.responsibility, axis=0))
            print ("np.sum(self.responsibility * np.square(self.w[:-1]), axis=0): ", np.sum(self.responsibility * np.square(self.w_array), axis=0))
            print ("division: ", np.sum(self.responsibility * np.square(self.w_array), axis=0) / np.sum(self.responsibility, axis=0))
        # update reg_lambda
        self.pi = (np.sum(self.responsibility, axis=0) + self.alpha - 1) / (self.w_array.shape[0] + self.gm_num * (self.alpha - 1))
        if step % self.gmuptfreq == 0:
            print ("reg_lambda", self.reg_lambda)
            print ("pi:", self.pi)

    def apply(self, trainnum, epoch, f, name, step):
        self.w_array = f.data.cpu().numpy().reshape((-1, 1)) # used for EM update also
        if epoch < 2 or step % self.paramuptfreq == 0:
            self.calcResponsibility()
            self.reg_grad_w = np.sum(self.responsibility*self.reg_lambda, axis=1).reshape(self.w_array.shape) * self.w_array
        reg_grad_w_dev = (torch.from_numpy((self.reg_grad_w.reshape(f.data.cpu().numpy().shape[0], -1))/float(trainnum))).float()
        if (epoch == 0 and step < 50) or step % self.gmuptfreq == 0:
            print ("step: ", step)
            print ("name: ", name)
            print ('data grad norm: ', np.linalg.norm(f.grad.data.cpu().numpy()))
            print ('reg_grad_w_dev l2 norm: ', np.linalg.norm(reg_grad_w_dev.cpu().numpy()))
        f.grad.data.add_(1.0, reg_grad_w_dev.cuda()) # here3
        if (epoch == 0 and step < 50) or step % self.gmuptfreq == 0:
            print ("delta w norm: ", np.linalg.norm(f.grad.data.cpu().numpy()))
            print ("w norm: ", np.linalg.norm(f.data.cpu().numpy()))
        if epoch < 2 or step % self.gmuptfreq == 0:
            if epoch >=2 and step % self.paramuptfreq != 0:
                self.calcResponsibility()
            self.update_GM_Prior_EM(name, step)

'''
class GMSGD(GMOptimizer, SGD):
    # The vallina Stochasitc Gradient Descent algorithm with momentum.
    # But this SGD has a GM regularizer

    def __init__(self, net=None, lr=None, momentum=None, weight_decay=None,
                 regularizer=None, constraint=None):
        GMOptimizer.__init__(self, net=net, lr=lr, momentum=momentum, weight_decay=weight_decay, regularizer=regularizer,
                                  constraint=constraint)
        SGD.__init__(self, lr=lr, momentum=momentum, weight_decay=weight_decay,
                 regularizer=regularizer, constraint=constraint)

    # compared with apply_with_lr, this need one more argument: isweight
    def apply_with_lr(self, dev, trainnum, net, epoch, lr, grad, value, name, step=-1):
        if grad.is_empty():
            return value
        ##### GM-prior: using gm_regularizer ##############
        grad = self.apply_GM_regularizer_constraint(dev=dev, trainnum=trainnum, net=net, epoch=epoch, value=value, grad=grad, name=name, step=step)
        ##### GM-prior: using gm_regularizer ##############
        if name is not None and name in self.learning_rate_multiplier:
            lr = lr * self.learning_rate_multiplier[name]
        self.opt.Apply(epoch, lr, name, grad.singa_tensor, value.singa_tensor)
        return value
'''
