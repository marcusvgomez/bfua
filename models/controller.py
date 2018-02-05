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
import torch.optim as optim

from models.agent import agent
from models.env import env, STATE_DIM
# from env import env, STATE_DIM


dtype = torch.FloatTensor

class Controller():
    def __init__(self, runtime_config):
        # params. Could probably make this a global constant instead of instance variable
        # but this works too
        self.N = runtime_config.num_agents
        self.M = runtime_config.num_landmarks
        self.K = runtime_config.vocab_size # vocabulary size
        
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
        # maybe X can just be a tensor and not a variable since it's not being trained
        # can X get its initial value from env?
        self.X = Variable(torch.randn(STATE_DIM*(self.N+self.M), self.N).type(dtype), requires_grad=False)
        
        self.C = Variable(torch.randn(self.K, self.N).type(dtype), requires_grad=True) # communication. one hot
        
        # create memory bank Tensors??
        self.M = None
        
        # create goals
        self.G = self.specify_goals()
        
        self.loss = self.compute_loss()
        
    def compute_loss(self):
        pass

    def specify_goals(self):
        # as the default just give some hardcoded goals that are useful for testing
        # perhaps allow for a flag/string param to allow for different goal sets
        # TODO: see if the second dimension is correct
        goals = Variable(torch.randn(self.N, self.GOAL_DIM).type(dtype), requires_grad=False)
        
        return goals
    
    def step(self):
        # get the policy action/comms from passing it through the agent network
        actions = self.agent.forward((self.X, self.C, self.g, self.M, self.m))
        next_state = self.env.forward(actions)
        self.X = next_state
        
    
    # # this is probably useless, ignore
    # def train(self, epochs=100):
    #     self.agent.train() # set to training mode
        
    #     optimizer = optim.SGD(self.agent.parameters(), lr=0.01)
    #     for _ in range(epochs): 
    #         # have some sort of training point (data, target)
    #         # get the next data point by passing to the environment
    #         data, target = Variable(data), Variable(target)
            
    #         optimizer.zero_grad()   # zero the gradient buffers
    #         output = self.agent(input_data)
    #         loss = self.compute_loss()
    #         loss.backward()
    #         optimizer.step()    # Does the update

def main():
    controller = Controller()
    
    # train it and feed in data?
    
if __name__ == '__main__':
    main()
