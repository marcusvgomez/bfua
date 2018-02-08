#our own modules
from models.agent import * 
from models.controller import *
from models.env import *
import argparse
from config import *

#torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Function, Variable
import torch.autograd as autograd

import torch.nn.init as init
from torch.nn.parameter import Parameter

#other imports
import numpy as np
import matplotlib.pyplot as plt

#model saving code
def save_model(model, optimizer, epoch_num, best_dev_acc, modelName, bestModelName, is_best = False):
    state = {'epoch': epoch_num + 1,
             'state_dict': model.state_dict(),
             'best_dev_acc': best_dev_acc,
             'optimizer': optimizer.state_dict()
            }
    torch.save(state, modelName)
    if is_best:
        shutil.copyfile(modelName, bestModel)

#updates the optimizer if we are going to do decay rates
def updateOptimizer(optimizer, decay_rate = 10):
    for param_group in self.optimizer.param_groups:
        param_group['lr'] /= decay_rate
        print (param_group['lr'])
    return optimizer

#plotting loss
def plot_loss(loss):
    x_axis = [i for i in range(len(loss))]
    y_axis = loss
    plt.plot(x_axis, y_axis, label = 'o')
    plt.xlabel = 'Epoch Number'
    plt.ylabel = 'Loss'
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Train time babbbyyyyyyyy")
    parser.add_argument('--n-epochs', '-e', type=int, help="Optional param specifying the number of epochs")
    parser.add_argument('--horizon', '-t', type=int, help="Optional param specifying the number of timesteps in a given epoch")
    parser.add_argument('--use_cuda', action='store_true', help="Optional param that specifies whether to train on cuda")
    parser.add_argument('--load-model', type=str, help='Optional param that specifies model weights to start using')
    parser.add_argument('--save-model', type=str, help='Optional param that specifies where to save model weights')
    parser.add_argument('--save-model-epoch', type=int, help='Optional param that specifies where to save model weights')
    parser.add_argument('--vocab-size', type=int, help='Optinoal param that specifies maximum vocabulary size')
    parser.add_argument('--num-agents', type=int, help='Optional param that specifies the number of agents')
    parser.add_argument('--num-landmarks', type=int, help='Optional param that specifies the number of landmarks')
    parser.add_argument('--hidden-comm-size', type=int, help='Optional param that specifies the number of hidden layers in comm')
    parser.add_argument('--hidden-input-size', type=int, help='Optional param that specifies the number of hidden layers in input')
    parser.add_argument('--hidden-output-size', type=int, help='Optional param that specifies the number of hidden layers in output')
    parser.add_argument('--learning-rate', type=float, help='Initial learning rate')
    parser.add_argument('--dropout', type=float, help='Dropout probability of neurons')
    parser.add_argument('--optimizer_decay', action='store_true', help='Number of epochs of not improving that we decay the optimizer')
    parser.add_argument('--optimizer-decay-epoch', type=float, help='Number of epochs of not improving that we decay the optimizer')
    parser.add_argument('--optimizer-decay-rate', type=float, help='Rate at which we decay the optimizer')
    parser.add_argument('--dirichlet-alpha', type=float, help='Optional param that specifies the Dirichlet Process hyperparameter used in communication reward')
    parser.add_argument('--deterministic-goals', type=bool, help='Optional param that specifies whether to generate a pre-specified dummy set of deterministic goals')

    arg_dict = vars(parser.parse_args())
    args = parser.parse_args()

    arg_dict['use_cuda'] = args.use_cuda


    runtime_config = RuntimeConfig(arg_dict)
    controller = Controller(runtime_config)

    #this needs to be fixed
    optimizer = optim.Adam(controller.agent.parameters(), lr = 0.01)
    assert(False) #this is here so stuff doesn't break

    loss = []
    not_improved = 0
    max_loss = float("-inf")
    for epoch in range(runtime_config.n_epochs):
        controller.reset()
        epoch_loss = []
        controller.run(runtime_confg.horizon)
        optimizer.zero_grad()
        total_loss = controller.compute_loss()
        total_loss.backward()
        optimizer.step()
        loss.append(total_loss)
        #only runs if we are using optimizer decay
        if total_loss < max_loss and args.optimizer_decay:
            max_loss = total_loss
            updateOptimizer(optimizer, runtime_config.optimizer_decay_rate)

    plot_loss(loss)





if __name__ == "__main__":
    main()

