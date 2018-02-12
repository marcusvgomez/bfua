'''
Agent implementation with no minibatching
'''

import sys
sys.path.append("../utils/")
from utils import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Function, Variable
import torch.autograd as autograd

import torch.nn.init as init
from torch.nn.parameter import Parameter

from torch.distributions import Categorical

from env import STATE_DIM, ACTION_DIM
'''
Agent operating in the environment 


num_agents: number of agents in the environment
vocab_size: size of the vocabulary we're operating on
num_landmarks: number of landmarks in the environment
input_size: size of the vector representing the representation of other landmarks/agents
hidden_comm_size: hidden communication size for FC communication layer
comm_output_size: size representing the output communication vector
hidden_input_size: size of the hidden layer for the location data
input_output_size: size of the output from the FC layers for location data
hidden_output_size: hidden size of the output layer for the FC layer
memory_size: size of the memory bank
goal_size: size of the goal
is_cuda: are we using cuda
'''
class agent(nn.Module):
    def __init__(self, num_agents, vocab_size, num_landmarks,
                 input_size, hidden_comm_size, comm_output_size,
                 hidden_input_size, input_output_size,
                 hidden_output_size, action_dim = ACTION_DIM,
                 memory_size = 32, goal_size = 6, is_cuda = False, dropout_prob = 0.0):
        super(agent, self).__init__()
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.vocab_size = vocab_size
        self.memory_size = memory_size
        self.input_size = input_size
        self.dropout_prob = dropout_prob
        self.action_dim = action_dim

        # print ("vocab size is: ", self.vocab_size + input_size)

        print "num agents and landmarks is: ", self.num_landmarks, self.num_agents

        self.communication_FC = nn.Sequential(
                nn.Linear(vocab_size + memory_size, hidden_comm_size),
                nn.ELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_comm_size, comm_output_size)
            )

        self.input_FC = nn.Sequential(
                nn.Linear(self.input_size * STATE_DIM, hidden_input_size),
                nn.ELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_input_size, input_output_size)
            )

        self.output_FC = nn.Sequential(
                nn.Linear(input_output_size + comm_output_size + goal_size + memory_size, hidden_output_size),
                nn.ELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_output_size, action_dim + vocab_size)
            )


        #activation functions and dropout
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax()
        self.log_softmax = nn.LogSoftmax()

        self.gumbel_softmax = GumbelSoftmax(tau=1.0,use_cuda = is_cuda)

        self.embeddings = nn.Embedding(vocab_size, vocab_size)

        if is_cuda:
            self.initializeCuda()

    '''
    Takes in inputs which is a tuple containing
    X: (N + M) x input_size matrix that represents the coordinates of other agents/landmarks
    C: N x communication_size matrix that represents the communication of all other agents
    g: goal vector always represented in R^3
    M: N x memory_size communcation memory matrix 
    m: state memory 
    Runs a forward pass of the neural network spitting out the action and communication actions
    '''
    def forward(self, inputs):
        X, C, g, M, m, is_training = inputs

        #reshaping everything for sanity
        M = M.transpose(1, 2)
        C = C.transpose(0, 1)
        X = X.transpose(0, 1)
        g = g.transpose(0, 1)
        m = m.transpose(0, 1)


        C = C.repeat(self.num_agents, 1, 1)
        X = X.repeat(self.num_agents, 1, 1)

        communication_input = torch.cat([C, M], 2) #concatenate along the first direction

        comm_out = self.communication_FC(communication_input)
        comm_pool = self.softmaxPool(comm_out)

        loc_output = self.input_FC(X)
        loc_pool = self.softmaxPool(loc_output, dim = 1).squeeze() #this is bad for now need to fix later


        #concatenation of pooled communication, location, goal, and memory
        output_input = torch.cat([comm_pool, m, loc_pool, g], 1)

        output = self.output_FC(output_input)

        psi_u, psi_c = output[:, :self.action_dim], output[:, self.action_dim:]
        
        action_output = psi_u + make_epsilon_noise()

        if is_training:
            communication_output = self.gumbel_softmax(psi_c)
        else:
            psi_c_log = self.softmax(psi_c)
            cat = Categorical(probs=psi_c_log)
            communication_output = cat.sample()

        #transposing because we have to i think
        #I really need to check to make sure math stuff works
        action_output = action_output.transpose(0,1)
        communication_output = communication_output.transpose(0,1)
       
        return action_output, communication_output

    '''
    Runs a softmax pool which is taking the softmax for all entries
    then returning the mean probabilities 
    '''
    def softmaxPool(self, inputs, dim = 0):
        input_prob = self.softmax(inputs)
        return torch.mean(input_prob, dim = dim)


    # I need to check this code
    def initializeCuda(self):
        # print "initializing Cuda"
        for param in self.parameter:
            print (param)



