'''
Controlla

Serves as the client that runs the RL agents to
facilitates their training (via computing reward 
and evaluating performance) and interaction with
the environment
'''

import numpy as np
import torch
from torch.autograd import Variable

from agent import agent
from env import env

dtype = torch.FloatTensor

class Controller():
    def __init__(self):
        # params. Could probably make this a global constant instead of instance variable
        # but this works too
        self.N = 12
        self.M = 3
        self.K = 20 # vocabulary size
        
        # the first 3 are one-hot for which action to perform go/look/nothing
        # Appendix 8.2: "Goal for agent i consists of an action to perform, a location to perform it on  r_bar, and an agent r that should perform that action"
        self.GOAL_DIM = 3 + 2 + 1
        
        self.env = env(num_agents=self.N, num_landmarks=self.M)
        
        # create agent policy network
        # OR it looks like create multiple since each network represents one agent?
        # or the agent network should just include all of possible numbers.
        # self.agents = [agent(...) for _ in range(N)]
        # self.agent = agent(num_agents=self.N, vocab_size=20, input_space=None, 
        #               hidden_comm_size=10, comm_output_size=50,
        #               input_size=100, hidden_input_size=20, 
        #               input_output_size=30, hidden_output_size=30)
        
        # each agent's observations of other agent/landmark locations from its 
        # own reference frame. TODO: probably not N+M for observations of all other objects
        self.X = Variable(torch.randn(self.N, self.N+self.M).type(dtype), requires_grad=False)
        self.C = Variable(torch.randn(self.N, self.K).type(dtype), requires_grad=True) # communication. one hot
        
        
        # create memory bank Tensors??
        self.M = None
        
        # create goals
        G = self.specify_goals()
        
    
    def set_landmark_states(self):
        pass
        
    def compute_reward(self):
        pass

    def specify_goals(self):
        # as the default just give some hardcoded goals that are useful for testing
        # perhaps allow for a flag/string param to allow for different goal sets
        pass


def main():
    controller = Controller()
    
if __name__ == '__main__':
    main()
