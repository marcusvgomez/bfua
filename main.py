import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append("./models")
sys.path.append("./utils")

#our own modules
import sys
sys.path.append("./utils/")
sys.path.append("models")
sys.path.append("../utils/")
# from utils.utils import *
from utils import *
from agent import * 
from controller import *
from env import *
from config import *
from visualize import draw

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
import argparse
import datetime
import shutil

import gc


def getTime():
    return datetime.datetime.now().strftime("%d-%H-%M-%s")

save_path = '/cvgl2/u/bcui/cs234/results/'
model_name = 'communication_reinforce'
currTime = getTime()
print currTime
save_model_name = save_path + model_name + " date " + currTime + ".pt"
best_name = save_path + "best/" + model_name + " date " + currTime + ".pt"
loss_dir = save_path + "loss/" + model_name + " date " + currTime + ".pt"


#model saving code
def save_model(model, optimizer, epoch_num, best_dev_acc, modelName = save_model_name, bestModelName = best_name, is_best = False):
    state = {'epoch': epoch_num + 1,
             'state_dict': model.state_dict(),
             'best_dev_acc': best_dev_acc,
             'optimizer': optimizer.state_dict()
            }
    torch.save(state, modelName)
    if is_best:
        shutil.copyfile(modelName, bestModelName)

#updates the optimizer if we are going to do decay rates
def updateOptimizer(optimizer, decay_rate = 5):
    print ("optimizing parameters")
    for param_group in optimizer.param_groups:
        param_group['lr'] /= decay_rate
        print (param_group['lr'])
    return optimizer

#plotting loss
def plot_loss(loss):
    x_axis = [i for i in range(len(loss))]
    y_axis = loss
    plt.plot(x_axis, y_axis, label = 'o')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.savefig('Loss_Deterministic_100Timesteps.png')

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

    # print (arg_dict['runtime-horizon'])
    runtime_config = RuntimeConfig(arg_dict)
    print (runtime_config.time_horizon)
    controller = Controller(runtime_config)

    #this needs to be fixed
    optimizer = optim.Adam(controller.agent_trainable.parameters(), lr = 0.0005)

    loss = []
    not_improved = 0
    min_loss = float("inf")
    save_loss = float("inf")
    # for epoch in range(runtime_config.n_epochs):
    # for epoch in range(int(1e6)):
    for epoch in range(5000):
        # for param in controller.agent_trainable.parameters():
            # print param


        controller.reset() #resetting the controller
        epoch_loss = []
        # controller.run(runtime_config.time_horizon)
        controller.run(10)
        optimizer.zero_grad()
        total_loss = controller.compute_loss()
        total_loss.backward()#retain_variables = True) #This code is sketchy at best, not sure what it does 
        optimizer.step()
        loss.append(total_loss.data[0])
        

        print "EPOCH IS: ", epoch, total_loss.data[0]
        # draw(controller.env.world_state_agents, 'vis' + str(epoch) + '.png')


        # if epoch % 50 == 0:
             # save_model(controller.agent_trainable, optimizer, epoch, min_loss, is_best = total_loss.data[0] < save_loss)
             # save_loss = min(save_loss, total_loss.data[0])

        #only runs if we are using optimizer decay
        # if total_loss.data[0] < max_loss and args.optimizer_decay:
        if total_loss.data[0] > min_loss:
            not_improved += 1
            if not_improved > 250:
                max_loss = total_loss
                optimizer = updateOptimizer(optimizer, runtime_config.optimizer_decay_rate)
                not_improved = 0
        else:
            min_loss = min(min_loss, total_loss.data[0])
            not_improved = 0

        del total_loss

        #this was done for memory checks
        # if epoch %10 == 0:
            # controller.reset()
            # check_memory()
        # if epoch == 1: assert False

    # print loss
    # with open(loss_dir, "wb") as f:
        # f.write(str(loss))


    plot_loss(loss)

def check_memory():
    for obj in gc.get_objects():
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            # print(type(obj), obj.size())
            del obj
    counter = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size())
            counter +=1
    print "number of parameters is: ", counter





if __name__ == "__main__":
    main()

