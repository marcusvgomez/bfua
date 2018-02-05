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




class agent(nn.Module):
	def __init__(self, num_agents, vocab_size,
				 input_size, hidden_comm_size, comm_output_size,
				 hidden_input_size, input_output_size,
				 hidden_output_size,
				 memory_size = 32, goal_size = 3, is_cuda = False, dropout_prob = 0.1):
		super(agent, self).__init__()
		self.num_agents = num_agents
		self.vocab_size = vocab_size
		self.memory_size = memory_size
		self.input_size = input_size
		self.dropout_prob = dropout_prob

		print ("vocab size is: ", self.vocab_size + input_size)

		self.com_FC1 = nn.Linear(vocab_size + memory_size, hidden_comm_size)
		self.com_FC2 = nn.Linear(hidden_comm_size, comm_output_size)

		self.input_FC1 = nn.Linear(input_size, hidden_input_size)
		self.input_FC2 = nn.Linear(hidden_input_size, input_output_size)

		print ("size is: ", input_output_size + comm_output_size + goal_size + memory_size)
		self.outputFC_1 = nn.Linear(input_output_size + comm_output_size + goal_size + memory_size, hidden_output_size)
		self.outputFC_2 = nn.Linear(hidden_output_size, input_size + vocab_size)

		#activation functions and dropout
		self.elu = nn.ELU()
		self.dropout = nn.Dropout(dropout_prob)
		self.softmax = nn.Softmax()

		self.gumbel_softmax = gumbel_softmax


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
	Runs a forward pass of the neural network spitting out the 
	'''
	def forward(self, inputs):
		X, C, g, M, m = inputs

		# nm, _ = X.shape

		communication_input = torch.cat([C, M], 1) #concatenate along the first direction

		hidden_comm = self.forwardFC(communication_input, self.elu, self.com_FC1, self.dropout)
		comm_out = self.forwardFC(hidden_comm, self.elu, self.com_FC2, self.dropout)
		comm_pool = self.softmaxPool(comm_out)


		loc_hidden = self.forwardFC(X, self.elu, self.input_FC1, self.dropout)
		loc_output = self.forwardFC(loc_hidden, self.elu, self.input_FC2, self.dropout)
		loc_pool = self.softmaxPool(loc_output)


		#concatenation of pooled communication, location, goal, and memory
		output_input = torch.cat([comm_pool, g, loc_pool, m], 0)
		output_hidden = self.forwardFC(output_input, self.elu, self.outputFC_1, self.dropout)
		output = self.forwardFC(output_hidden, self.elu, self.outputFC_2, self.dropout)


		psi_u, psi_c = output[:self.input_size], output[self.input_size:]
		print (psi_u, psi_c)
		psi_c_log = self.softmax(psi_c)
		# psi_c_gumbel = self.gumbel_softmax(psi_c)


		m = Categorical(psi_c_log)
		c_action = m.sample()

		# return c_action

		communication_output = self.embeddings(c_action)

		return communication_output

	'''
	Runs a softmax pool which is taking the softmax for all entries
	then returning the mean probabilities 
	'''
	def softmaxPool(self, inputs):
		input_prob = self.softmax(inputs)
		return torch.mean(input_prob, 0)


	def forwardFC(self, currInput, activation, layer, dropout = None):
		if dropout is None:
			return activation(layer(currInput))
		else:
			return dropout(activation(layer(currInput)))



	# I need to check this code
	def initializeCuda(self):
		# print "initializing Cuda"
		for param in self.parameter:
			print (param)



