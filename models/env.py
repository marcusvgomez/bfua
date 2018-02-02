'''
Environment implementation

Takes as input the actions of each agent at a given time step
and outputs the observation of the state of the world from 
the perspective of each agent. Internally maintains the current
state of the world.
'''
import torch

R_DIM = 3
STATE_DIM = R_DIM * 4# 3(for position) + 3(for velocity) + 3(for gaze) + 3(for color)
ACTION_DIM = R_DIM * 2 # 3 (for velocity) + 3(for gaze)

class env:
    def __init__(self, num_agents, num_landmarks, timestep=0.1, damping_coef=0.5):
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.timestep = timestep #delta t
        self.damping_coef = damping_coef # this is 1 - gamma
        self.gamma = 1 - self.damping_coef
        self.world_state_agents = torch.FloatTensor(STATE_DIM, self.num_agents).zero_()
        self.world_state_landmarks = torch.FloatTensor(STATE_DIM, self.num_landmarks).zero_()
        
        ## this probably shouldn't be hardcoded but...whatever
        self.transform_L = torch.FloatTensor(STATE_DIM, STATE_DIM)
        self.transform_L[0:3,0:3] = torch.eye(R_DIM)
        self.transform_L[0:3,3:6] = self.timestep * torch.eye(R_DIM)
        self.transform_L[3:6,3:6] = self.gamma * torch.eye(R_DIM)
        self.transform_L[9:12,9:12] = torch.eye(R_DIM)
        self.transform_R = torch.FloatTensor(STATE_DIM, ACTION_DIM)
        self.transform_R[3:6,0:3] = self.timestep * torch.eye(R_DIM)
        self.transform_R[6:9,3:6] = torch.eye(R_DIM)

    def modify_world_state(self, agents, landmarks):
        pass

    ##actions should be a 6 X N tensor
    ##returns a 12(N+M) x N tensor
    def forward(self, actions):
        L = torch.matmul(self.transform_L, self.world_state_agents)
        R = torch.matmul(self.transform_R, actions)
        self.world_state_agents = L + R
        result = torch.FloatTensor(STATE_DIM*(self.num_agents + num_landmarks), self.num_agents)
        for i in range(self.num_agents):
            row = torch.FloatTensor(STATE_DIM*(self.num_agents + num_landmarks))
            for j in range(self.num_agents):
                row[STATE_DIM*j:STATE_DIM*(j+1)] = self.world_state_agents[:,j] - self.world_state_agents[:,i]
                row[STATE_DIM*j + 9] = self.world_state_agents[9,j]
                row[STATE_DIM*j + 10] = self.world_state_agents[10,j]
                row[STATE_DIM*j + 11] = self.world_state_agents[11,j]
            offset = STATE_DIM*self.num_agents
            for j in range(self.num_landmarks):
                row[offset + STATE_DIM*j: offset + STATE_DIM*(j+1)] = self.world_state_landmarks[:,j] - self.world_state_agents[:,i]
                row[offset + STATE_DIM*j + 6] = 0.0
                row[offset + STATE_DIM*j + 7] = 0.0
                row[offset + STATE_DIM*j + 8] = 0.0
                row[offset + STATE_DIM*j + 9] = self.world_state_landmarks[9,j]
                row[offset + STATE_DIM*j + 10] = self.world_state_landmarks[10,j]
                row[offset + STATE_DIM*j + 11] = self.world_state_landmarks[11,j]
            result[:,i] = row
        return result

        
