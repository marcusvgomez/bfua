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

from agent import agent
from env import env, STATE_DIM

dtype = torch.FloatTensor

GOAL_DIM = 3 + 2 + 1

class Controller():
    def __init__(self, runtime_config):
        # params. Could probably make this a global constant instead of instance variable
        # but this works too
        self.N = runtime_config.num_agents
        self.M = runtime_config.num_landmarks
        self.K = runtime_config.vocab_size # vocabulary size
        self.dirichlet_alpha = runtime_config.dirichlet_alpha
        
        # the first 3 are one-hot for which action to perform go/look/nothing
        # Appendix 8.2: "Goal for agent i consists of an action to perform, a location to perform it on  r_bar, and an agent r that should perform that action"
        
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
        
        # keeps track of the utterances 
        self.comm_counts = [0] * self.K
    
    def compute_prediction_loss(self):
        # can only be completed once the agent network is also predicting goals
        return 0
    
    def compute_comm_loss(self):
        # for penalizing large vocabulary sizes
        r_c = 0
        n = sum(self.comm_counts)
        probs = [self.comm_counts[i] / (self.dirichlet_alpha + n - 1.0) for i in range(self.K)]
        # TODO: finish this and store the entire record to compute dirichlet process
        return -r_c
    
    def compute_loss(self):
        # TODO: fill in these rewards. Physical will come from env.
        physical_loss = self.compute_physical_loss(self.G)
        prediction_loss = self.compute_prediction_loss()
        comm_loss = self.compute_comm_loss()
        self.loss = -(physical_loss + prediction_loss + comm_loss)
        return self.loss

    def specify_goals(self):
        # as the default just give some hardcoded goals that are useful for testing
        # perhaps allow for a flag/string param to allow for different goal sets
        goals = Variable(torch.randn(GOAL_DIM, self.N).type(dtype), requires_grad=False)
        
        return goals
    
    def update_comm_counts(self):
        # update counts for each communication utterance. counts are used in comm_reward 
        comms = np.argmax(self.C.data.numpy(), axis=0).ravel()
        for c in comms:
            comm_counts[c] += 1
    
    def step(self):
        # get the policy action/comms from passing it through the agent network
        actions, self.C = self.agent.forward((self.X, self.C, self.g, self.M, self.m))
        self.X = self.env.forward(actions)
        
        self.update_comm_counts()
        

def main():
    controller = Controller()
    
if __name__ == '__main__':
    main()
