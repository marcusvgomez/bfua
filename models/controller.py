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
from models.env import env, STATE_DIM, ACTION_DIM
# from env import env, STATE_DIM


import sys
sys.path.append("./utils/")
from utils import *


dtype = torch.FloatTensor

GOAL_DIM = 3 + 2 + 1

class Controller():
    def __init__(self, runtime_config):
        # params. Could probably make this a global constant instead of instance variable
        # but this works too
        self.N = runtime_config.num_agents
        self.M = runtime_config.num_landmarks
        self.K = runtime_config.vocab_size # vocabulary size


        self.hidden_comm_size = runtime_config.hidden_comm_size
        self.hidden_input_size = runtime_config.hidden_input_size
        self.hidden_output_size = runtime_config.hidden_output_size
        self.comm_output_size = runtime_config.comm_output_size
        self.input_output_size = runtime_config.input_output_size

        self.dirichlet_alpha = runtime_config.dirichlet_alpha
        self.deterministic_goals = runtime_config.deterministic_goals
        
        # the first 3 are one-hot for which action to perform go/look/nothing
        # Appendix 8.2: "Goal for agent i consists of an action to perform, a location to perform it on  r_bar, and an agent r that should perform that action"
        
        self.env = env(num_agents=self.N, num_landmarks=self.M)
        
        # create agent policy network
        # OR it looks like create multiple since each network represents one agent?
        # or the agent network should just include all of possible numbers.
        # self.agents = [agent(...) for _ in range(N)]
        self.agent = agent(num_agents=self.N, vocab_size=self.K, num_landmarks=self.M,
                 input_size=self.N+self.M, hidden_comm_size=self.hidden_comm_size, 
                 comm_output_size = self.comm_output_size,
                 hidden_input_size=self.hidden_input_size, input_output_size=self.input_output_size,
                 hidden_output_size=self.hidden_output_size,
                 memory_size = 32, goal_size = GOAL_DIM, is_cuda = False, dropout_prob = 0.1)
        
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
        
        #self.loss = self.compute_loss()
        self.physical_losses = []
        
        # keeps track of the utterances 
        self.comm_counts = torch.Tensor(self.K).type(dtype)
    
    def reset(self):
        self.env = env(num_agents=self.N, num_landmarks=self.M)
        self.G = self.specify_goals()
        self.physical_losses = []
        self.comm_counts = torch.Tensor(self.K).type(dtype)

   
    def compute_prediction_loss(self):
        # can only be completed once the agent network is also predicting goals
        return 0
    
    def compute_comm_loss(self):
        # for penalizing large vocabulary sizes
        # probs should all be greater than 0
        probs = self.comm_counts / (self.dirichlet_alpha + torch.sum(self.comm_counts) - 1.)
        r_c = torch.sum(torch.cmul(self.comm_counts, torch.log(probs)))
        return -r_c
    
    def compute_physical_loss(self):
        return sum(self.physical_losses)
        
    def compute_loss(self):
        # TODO: fill in these rewards. Physical will come from env.
        physical_loss = self.compute_physical_loss()
        prediction_loss = self.compute_prediction_loss()
        comm_loss = self.compute_comm_loss()
        self.loss = -(physical_loss + prediction_loss + comm_loss)
        return self.loss

    def specify_goals(self):
        # if runtime param wants deterministic goals, then we manually specify a set of simple
        # goals to train on
        # Goals are formatted as 6-dim vectors: [one hot action selection, location coords, agent] (3 + 2 + 1)
        # Otherwise, randomly generate one
        
        goals = torch.FloatTensor(GOAL_DIM, self.N).zero_()
        if self.deterministic_goals:
            # agent 0's goal is to get agent 1 to go to (5, 5)
            goals[:, 0] = torch.FloatTensor([1, 0, 0, 5, 5, 1])
            # agent 1's goal is to get agent 0 to look UP at (0, 1)
            goals[:, 1] = torch.FloatTensor([0, 1, 0, 0, 1, 0])
            # agent 2's goal is to send itself to (-5, -5)
            goals[:, 2] = torch.FloatTensor([1, 0, 0, -5, -5, 2])
            # the rest just do nothing
            for i in range(2, N):
                goals[2, i] = 1
        else:
            for i in range(N):
                action_type = np.random.randint(0, 3) # either go-to, look-at, or do-nothing
                x, y = np.random.uniform(-20.0, 20.0, size=(2,)) # TODO: have clearer bounds in env so these coordinates mean something
                target_agent = np.random.randint(0, N)
                
                goals[i, action_type] = 1
                goals[i, 3] = x
                goals[i, 4] = y
                goals[i, 5] = target_agent

        return goals
    
    def update_comm_counts(self):
        # update counts for each communication utterance.
        # interpolated as a float for differentiability, used in comm_reward 
        comms = torch.sum(self.C, axis=1)
        self.comm_counts += comms
    
    def update_phys_loss(self, actions):
        world_state_agents, world_state_landmarks = self.env.expose_world_state()
        goals = self.G ## GOAL_DIM x N
        loss_t = 0.0
        for i in range(self.N):
            g_a_i = goals[:,i]
            g_a_r = g_a_i[GOAL_DIM - 1]
            r_bar = g_a_i[3:5]
            if g_a_i[0] == 1:
                p_t_r = world_state_agents[:,g_a_r][0:2]
                loss_t += (p_t_r - r_bar).norm(2)
            else if g_a_i[1] == 1: 
                v_t_r = world_state_agents[:,g_a_r][4:6]
                loss_t += (v_t_r - r_bar).norm(2)
            u_i_t = actions[:,i]
            c_i_t = self.C[:,i]
            loss_t += u_i_t.norm(2)
            loss_t += c_i_t.norm(2)
        loss_t *= -1.0
        self.physical_losses.append(loss_t)

    def step(self):
        # get the policy action/comms from passing it through the agent network
        actions, self.C = self.agent.forward((self.X, self.C, self.g, self.M, self.m))
        self.X = self.env.forward(actions)
        
        self.update_comm_counts()
        self.update_phys_loss(actions)
    
    def run(self, t):
        for iter_ in len(t):
            self.step()

        

def main():
    controller = Controller()
    
if __name__ == '__main__':
    main()
