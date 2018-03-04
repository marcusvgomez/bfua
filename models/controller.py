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
from env import env, STATE_DIM, ACTION_DIM
from visualize import draw
# from env import env, STATE_DIM

import time
import sys
import os
sys.path.append("./utils/")
from utils import *

dtype = torch.FloatTensor

GOAL_DIM = 3 + 2 + 1

class Controller():
    def __init__(self, runtime_config):
        self.GLOBAL_ITER = 0
        # params. Could probably make this a global constant instead of instance variable
        # but this works too
        self.N = runtime_config.num_agents
        self.M = runtime_config.num_landmarks
        self.K = runtime_config.vocab_size # vocabulary size

        self.memory_size = runtime_config.memory_size
        self.hidden_comm_size = runtime_config.hidden_comm_size
        self.hidden_input_size = runtime_config.hidden_input_size
        self.hidden_output_size = runtime_config.hidden_output_size
        self.comm_output_size = runtime_config.comm_output_size
        self.input_output_size = runtime_config.input_output_size


        self.dirichlet_alpha = runtime_config.dirichlet_alpha
        self.deterministic_goals = runtime_config.deterministic_goals

        self.runtime_config = runtime_config
        
        # the first 3 are one-hot for which action to perform go/look/nothing
        # Appendix 8.2: "Goal for agent i consists of an action to perform, a location to perform it on  r_bar, and an agent r that should perform that action"
        
        self.env = env(num_agents=self.N, num_landmarks=self.M, is_cuda=self.runtime_config.use_cuda)
        
        # create agent policy network
        # OR it looks like create multiple since each network represents one agent?
        # or the agent network should just include all of possible numbers.
        # self.agents = [agent(...) for _ in range(N)]
        self.agent_trainable = agent(num_agents=self.N, vocab_size=self.K, num_landmarks=self.M,
                 input_size=self.N+self.M, hidden_comm_size=self.hidden_comm_size, 
                 comm_output_size = self.comm_output_size,
                 hidden_input_size=self.hidden_input_size, input_output_size=self.input_output_size,
                 hidden_output_size=self.hidden_output_size,
                 memory_size = 32, goal_size = GOAL_DIM, is_cuda = runtime_config.use_cuda, dropout_prob = 0.1,
                 is_goal_predicting = True)


        # each agent's observations of other agent/landmark locations from its 
        # own reference frame. TODO: probably not N+M for observations of all other objects
        # maybe X can just be a tensor and not a variable since it's not being trained
        # can X get its initial value from env?
        self.X = Variable(torch.randn(STATE_DIM*(self.N+self.M), self.N).type(dtype), requires_grad=True)
        
        self.C = Variable(torch.zeros(self.K, self.N).type(dtype), requires_grad=True) # communication. one hot
        self.G_loss = 0.0 
        # create memory bank Tensors??
        self.Mem = Variable(torch.zeros(self.N, self.memory_size, self.N).type(dtype), requires_grad = True)
        
        # create goals
        self.G = self.specify_goals()
        
        self.physical_losses = []
        
        # keeps track of the utterances 

        #TODO: Need to properly implement memory
        self.mem = Variable(torch.zeros(self.memory_size, self.N).type(dtype), requires_grad = True)
        self.comm_counts = Variable(torch.zeros(self.K).type(dtype), requires_grad = True)
        
        if runtime_config.use_cuda:
            print "running cuda"
            self.agent_trainable.cuda()
            self.comm_counts = self.comm_counts.cuda()

        print "running deterministic goals: ", self.deterministic_goals
        
        # make directory for storing visualizations
        self.img_dir = os.path.dirname(__file__) + '/../imgs/' + "phys_comm" + time.strftime('%m%d-%I%M') + '/'
        try:
            os.mkdir(self.img_dir[:-1])
        except OSError as err:
            pass
    
    def reset(self):
        del self.physical_losses[:]
        self.env.clear()
        del self.G
        self.G_loss = 0.0

        self.env = env(num_agents=self.N, num_landmarks=self.M, is_cuda=self.runtime_config.use_cuda)
        self.Mem = Variable(torch.zeros(self.N, self.memory_size, self.N).type(dtype), requires_grad = True)
        self.mem = Variable(torch.zeros(self.memory_size, self.N).type(dtype), requires_grad = True)
        self.G = self.specify_goals()
        
        self.physical_losses = []
        self.comm_counts = torch.zeros(self.K).type(dtype)
        self.C = Variable(torch.zeros(self.K, self.N).type(dtype), requires_grad=True) # communication. one hot
        if self.runtime_config.use_cuda:
            self.Mem = self.Mem.cuda()
            self.mem = self.mem.cuda()
            self.G = self.G.cuda()
            self.comm_counts = self.comm_counts.cuda()
            self.C = self.C.cuda()


    ##predictions are N x goal x N
    def update_prediction_loss(self, predictions):
        goals = self.G ## goal x N
        ret = 0.0
        for i in range(self.N):
            for j in range(self.N):
                if i == j: continue
                i_prediction_j = predictions[i,:,j]
                j_true = goals[:,j]
                ret += torch.norm(i_prediction_j - j_true)
        self.G_loss =  -1.0 * ret
    
    def compute_prediction_loss(self): return self.G_loss

    def compute_comm_loss(self):
        # for penalizing large vocabulary sizes
        # probs should all be greater than 
        probs = self.comm_counts / (self.dirichlet_alpha + torch.sum(self.comm_counts) - 1.)
        r_c = torch.sum(self.comm_counts * torch.log(probs))
        # print "comm reward is: ", r_c
        return -r_c
    
    def compute_physical_loss(self):
        return sum(self.physical_losses)
        
    def compute_loss(self):
        # TODO: fill in these rewards. Physical will come from env.
        physical_loss = self.compute_physical_loss()
        # print "physical loss is: ", physical_loss.data[0]
        prediction_loss = self.compute_prediction_loss()
        comm_loss = self.compute_comm_loss()
        # prediction_loss = 0.
        # comm_loss = 0.
        self.loss = -(physical_loss + prediction_loss + comm_loss)
        return self.loss

    def specify_goals(self):
        # if runtime param wants deterministic goals, then we manually specify a set of simple
        # goals to train on
        # Goals are formatted as 6-dim vectors: [one hot action selection, location coords, agent] (3 + 2 + 1)
        # Otherwise, randomly generate one
        
        goals = torch.FloatTensor(GOAL_DIM, self.N).zero_()
        if self.deterministic_goals:
            # ACTUALLY rn agent 0 is just doing to do nothing. simplest case for now. agent 0's old goal is to get agent 1 to go to (5, 5)
            # goals[:, 0] = torch.FloatTensor([0, 0, 1, 5, 5, 1])
            # # ACTUALLY rn agent 1 goal is also to do nothing. agent 1's old goal is to get agent 0 to look UP at (0, 1)
            # goals[:, 1] = torch.FloatTensor([0, 0, 1, 0, 1, 0])
            #agent 0's goal is to get agent 1 to go to (5,5)
            goals[:, 0] = torch.FloatTensor([0, 0, 1, 5, 5, 1])
            #agent 1's goal is to get agent 0 to look UP at (0,1)
            goals[:, 1] = torch.FloatTensor([0, 1, 0, 5, -5, 0])
            # agent 2's goal is to send itself to (-5, -5)
            goals[:, 2] = torch.FloatTensor([0, 0, 1, -5, -5, 2])
            # the rest just do nothing
            for i in range(3, self.N):
                goals[2, i] = 1
        else:
            for i in range(self.N):
                action_type = np.random.randint(0, 3) # either go-to, look-at, or do-nothing
                x, y = np.random.uniform(-20.0, 20.0, size=(2,)) # TODO: have clearer bounds in env so these coordinates mean something
                target_agent = np.random.randint(0, self.N)
                
                goals[action_type,i] = 1
                goals[3,i] = x
                goals[4,i] = y
                goals[5,i] = target_agent

        return Variable(goals.type(dtype), requires_grad = True)
    
    def update_comm_counts(self):
        # update counts for each communication utterance.
        # interpolated as a float for differentiability, used in comm_reward 

        comms = torch.sum(self.C, dim=1)
        self.comm_counts += comms.data
    
    def update_phys_loss(self, actions):
        world_state_agents, world_state_landmarks = self.env.expose_world_state()
        goals = self.G ## GOAL_DIM x N
        loss_t = 0.0
        # loss_t = Variable(torch.FloatTensor([0.]))
        for i in range(self.N):
            # if i != 2: continue
            g_a_i = goals[:,i]
            g_a_r = int(g_a_i[GOAL_DIM - 1].data[0])
            r_bar = g_a_i[3:5]
            ## GOTO action
            if g_a_i.data[0] == 1:
                p_t_r = world_state_agents[:,g_a_r][0:2]
                try:
                    loss_t += ((p_t_r - r_bar).norm(2))**2
                except RuntimeError as e:
                    pass
                    # print "FUCK", p_t_r, r_bar
            #GAZE action
            elif g_a_i.data[1] == 1: 
                v_t_r = world_state_agents[:,g_a_r][4:6]
                try:
                    loss_t += ((v_t_r - r_bar).norm(2))**2
                except RuntimeError as e:
                    pass
                    # print "FUCK", v_t_r, 
            u_i_t = actions[:,i]
            c_i_t = self.C[:,i]
            # loss_t += u_i_t.norm(2) * 0.005
            # loss_t += c_i_t.norm(2) * 0.005
        loss_t *= -1.0
        self.physical_losses.append(loss_t)

    def step(self, is_training = True, debug=False):
        # get the policy action/comms from passing it through the agent network
        if self.runtime_config.use_cuda:
            self.X = self.X.cuda()
            self.C = self.C.cuda()
            self.G = self.G.cuda()
            self.Mem = self.Mem.cuda()
            self.mem = self.mem.cuda()


        actions, cTemp, MemTemp, memTemp, goal_out = self.agent_trainable((self.X, self.C, self.G, self.Mem, self.mem, is_training))
        self.update_phys_loss(actions)

        #not sure where 
        # actions = Variable(actions.data, requires_grad = False)
        self.Mem = Variable(MemTemp.data, requires_grad = True)
        self.mem = Variable(memTemp.data, requires_grad = True)
        self.C = Variable(cTemp.data, requires_grad = True)
        #self.G = Variable(goal_out.data, requires_grad = True)

        tempX = self.env.forward(actions)
        self.X = Variable(tempX.data, requires_grad = True)


        self.GLOBAL_ITER += 1
        self.update_prediction_loss(goal_out)
        self.update_comm_counts()
        if debug and self.GLOBAL_ITER % 100 == 0: print actions
    
    def run(self, t):
        # self.GLOBAL_ITER += 1
        # print self.GLOBAL_ITER
        for iter_ in range(t):
    #        print self.img_dir
            if iter_ == t - 1: 
                # if self.GLOBAL_ITER % 100 == 99:
                    # print self.env.expose_world_state()[0]
                self.step(debug=False)
            else:
                self.step()

            
            # visualize every 10 time steps
            if self.GLOBAL_ITER % 10 == 0:
                draw(self.env.world_state_agents, name=self.img_dir + 'vis'+str(self.GLOBAL_ITER)+ '_' + '.png')
        # if self.GLOBAL_ITER == 10000:
            # assert False

        

def main():
    controller = Controller()
    
if __name__ == '__main__':
    main()
