import os
import sys 
import torch
import torch.nn as nn 
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 

import matplotlib
import matplotlib.pyplot as plt 

import math 
import numpy as np 

# find the current path
path = os.path.dirname( os.path.abspath( __file__))

class RBM( nn.Module):
    '''Restricted Botlzmann Machine

    Reference: https://github.com/mehulrastogi/Deep-Belief-Network-pytorch
    '''
    def __init__( self, nV, nH, 
                        k=3, 
                        if_ada_k=True,
                        if_gpu=False,
                        verbose=True,
                        beta1=.9,
                        beta2=.99,
                        lr=1e-4,):
        super().__init__()
        self.nV = nV
        self.nH = nH
        self.k_samples = k 
        self.if_ada_k = if_ada_k
        dev = 'cuda' if torch.cuda.is_available() and if_gpu else 'cpu'
        self.dev = torch.device( dev)
        self.verbose = verbose
        self._init_RBM()
        self._init_optimizer( beta1=beta1, beta2=beta2, lr=lr)

    def _init_RBM( self):
        '''Init the param
        W ~ N(0,0.01)
        b = 0 
        c = 0 
        '''
        self.W = torch.normal( mean=0, std=.01, 
                    size=(self.nV,self.nH)).to(self.dev)
        self.b = torch.zeros( size=(1,self.nV), 
                    dtype=torch.float32).to(self.dev)
        self.c = torch.zeros( size=(1,self.nH), 
                    dtype=torch.float32).to(self.dev)
    
    def _init_optimizer( self, beta1=.9, beta2=.99, lr=1e-4, eps=1e-7):
        '''Init Adam
            init momentum: m
            init history:  r
        '''
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr    = lr 
        self.eps   = eps 
        self.m     = { 'W': 0, 'b':0, 'c':0}
        self.r     = { 'W': 0, 'b':0, 'c':0}
        self.l2    = 1e-4
        
    def to_hidden( self, x):
        '''Sample from p(h|v)
        x ~ v
        E(h|v) = - x @ W - c
        p(h|v) ∝ exp( -E(v|h))
        '''
        e = - torch.mm( x, self.W) - self.c
        p_h1v = torch.sigmoid( -e)
        return p_h1v, torch.bernoulli( p_h1v)

    def to_visible( self, y):
        '''Sample from p(v|h)
        y ~ h
        E(v|h) = - y @ w - b
        p(v|h) ∝ exp( -E(v|h))
        '''
        e = - torch.mm( y, self.W.t()) - self.b 
        p_v1h = torch.sigmoid( -e)
        return p_v1h, torch.bernoulli( p_v1h)

    def adam( self, grad, idx):
        '''
        Reference:
        https://www.zhihu.com/question/323747423

        m = β1 * m + (1 - β1) * grad
        r = β2 * r + (1 - β2) * grad ⊙ grad 
        lr = lr * m / (sqrt(r) + epsilon)
        '''
        # update momentum
        self.m[ idx] = self.beta1*self.m[ idx] \
                        + ( 1. - self.beta1)*grad
        # update history 
        self.r[ idx] = self.beta2*self.r[ idx] \
                        + ( 1. - self.beta2)*grad.pow(2)
    
        return self.m[ idx] / ( np.sqrt(self.r[ idx]) + self.eps)

    def step( self, x, train=True, n_sample=1, lr=1e-4):
        '''
        '''
        ## Inferece
        # Positive phase: Bottom-up
        p_h1v_pos, h = self.to_hidden( x)
        vh_pos = torch.mm( x.t(), h)
        # Negative phase: Top-down 
        if self.if_ada_k:
            n_sample = int( np.ceil( ((self.epoch+1)/(self.n_epoch+1)) * self.k_samples))
        else: 
            n_sample = self.k_samples
        for _ in range(n_sample):
            p_v1h, _ = self.to_visible( h)
            p_h1v_neg, h = self.to_hidden( p_v1h)
        vh_neg = torch.mm( p_v1h.t(), p_h1v_neg)

        ## Learning 
        W_grad = 0
        if train:
            # △W = (v+h+ - v-h-) + l2*2*w
            W_grad = (( vh_pos - vh_neg) + self.l2*2*self.W
                        ) / self.batch_size 
            b_grad = (( x - p_v1h).sum(dim=0) + self.l2*2*self.b
                        ) / self.batch_size
            c_grad = (( p_h1v_pos - p_h1v_neg).sum(dim=0) + self.l2*2*self.c
                        ) / self.batch_size
            # update 
            self.W += lr * self.adam( W_grad, 'W')
            self.b += lr * self.adam( b_grad, 'b')
            self.c += lr * self.adam( c_grad, 'c')
        
        ## Computer reconstruction error 
        err = (x - p_v1h).pow(2).sum(dim=0).mean()

        return err, torch.norm(W_grad)

    def train( self, train_data, n_epoch, batch_size):
        
        ## get batch_size
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        losses = []
        
        # start training
        for self.epoch in range( n_epoch):
            
            n_batch = int( len(train_data))
            cost_ = torch.FloatTensor( n_batch, 1)
            grad_ = torch.FloatTensor( n_batch, 1)

            ## train each batch 
            for i, (x_batch, _) in tqdm(enumerate(train_data), ascii=True,
                                desc='RBM fitting', file=sys.stdout):
                x_batch = x_batch.view( [-1, self.nV])
                cost_[i-1], grad_[i-1] = self.step( x_batch, True, self.epoch, n_epoch)

            ## Track learning
            if (self.epoch % 2 == 0) and self.verbose:
                print( f'Epoch:{self.epoch}, avg_loss={cost_.mean()}, avg_grad={grad_.mean()} ')

            # Save the loss
            losses.append( cost_.mean().numpy())

        return losses

class DBN( nn.Module):
    '''Deep Belief Network

    Reference: https://github.com/mehulrastogi/Deep-Belief-Network-pytorch
    '''
    def __init__( self, dims,
                        k=3, 
                        if_ada_k=True,
                        if_gpu=False,
                        verbose=True,
                        beta1=.9,
                        beta2=.99,
                        lr=1e-4,):
        super( DBN, self).__init__()
        # stack RBMs to construct DBN
        self.RBM_layers = []
        for i in range(len(dims)-1):
            rbm = RBM( dims[i], dims[i+1],
                        k=k, 
                        if_ada_k=if_ada_k,
                        if_gpu=if_gpu,
                        verbose=verbose,
                        beta1=beta1,
                        beta2=beta2,
                        lr=lr,)
            self.RBM_layers.append( rbm)
    
    def forward( self, x):
        '''
        v_: the hidden node in the last layer is the visible
            node for the next 
        '''
        v_ = x 
        for i in range( len(self.RBM_layers)):
            v = v_.view( [v_.shape[0], -1]).type( torch.FloatTensor)
            p_h1v, v_ = self.RBM_layers[i].to_hidden(v)
        return p_h1v, v_ 

    def perceive( self, x):
        '''
        Perception process: also called reconstruct 
        Bottom-up and then top-down
        '''
        ## Bottom-up 
        h = x 
        for i in range( len(self.RBM_layers)):
            v = h.view( [h.shape[0], -1]).type( torch.FloatTensor)
            _, h = self.RBM_layers[i].to_hidden(v)
        
        ## Top-down 
        v = h 
        for i in range( len(self.RBM_layers)):
            h = v.view( [v.shape[0], -1]).type( torch.FloatTensor)
            p_v1h, v = self.RBM_layers[i].to_visible(h)

        p_v1x = p_v1h
        return p_v1x, v    
        
    def train_static( self, train_data, train_labels, 
                            n_epoch=50, batch_size=32):
        '''
        Greedy Layer By layer training
        Can be used for pre-train
        '''
        h = train_data 
        train_hist = dict()
        for i in range(len(self.RBM_layers)):

            # track the training 
            print( f"{'-'*20}\nTraining the {i+1} layer")

            # load the data with torch dataloader 
            x_tensor = h.type( torch.FloatTensor)
            y_tensor = train_labels.type( torch.FloatTensor)
            _dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
            _dataloader = torch.utils.data.DataLoader( _dataset, 
                        batch_size=batch_size, drop_last=True)
                        
            # train each layer 
            loss = self.RBM_layers[i].train( _dataloader, n_epoch, batch_size)
            v = h.view( [h.shape[0], -1]).type(torch.FloatTensor)
            _, h = self.RBM_layers[i].to_hidden( v)
            v = h 

            # save for visualization
            train_hist[f'layer{i+1}'] = loss 
        
        ## Save the network parameters 
        torch.save(self.RBM_layers.state_dict(), f'{path}/checkpts/DBN_params.pkl')

    def load_state( self, path):
        self.RBM_layers.load_state_dict(torch.load(path)) 
          
if __name__ == '__main__':

    ## Load  MNIST dataset 
    mnist_data = datasets.MNIST('../data', train=True, download=True,
                                transform=transforms.Compose(
                                    [ transforms.ToTensor(), transforms.Normalize( (.1307,), (.3081,))]
                                ))
    data = (mnist_data.data.type( torch.FloatTensor) / 255).bernoulli()
    label = (mnist_data.targets.type( torch.FloatTensor) / 255).bernoulli()

    ## Init the model 
    dbn = DBN( [ 784, 500, 500, 2000])

    ## train the model
    n_epoch = 20
    batch_size = 32
    dbn.train_static( data, label, n_epoch, batch_size)

    ## 
    