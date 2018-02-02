import numpy as np
import torch

import torch.nn.functional as F
from torch.autograd import Variable

'''
Takes log probabilities and outputs and outputs the gummel softmax for sampling
Taken from:
https://github.com/pytorch/pytorch/issues/639
'''
def gumbel_softmax(prob, tau = 1):
	noise = Variable(make_gummel_noise())
	x = (input + noise) / tau
	x = F.softmax(x.view(input.size(0), -1))
	return x.view_as(input)

'''
Makes gummel noise as described in the paper
'''
def make_gummel_noise():
	noise = torch.rand(input.size())
	noise.add_(1e-9).log_().neg_()
	noise.add_(1e-9).log_().neg_()
	return noise