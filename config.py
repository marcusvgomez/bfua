"""
Training config

Contains runtime config params
"""

class RuntimeConfig:

    def __init__(self, args):
        print (args)
        self.using_cuda = args['use_cuda']
        self.n_epochs = 10000 if 'n-epochs' not in args else args['n-epochs']
        self.time_horizon = 1000 if 'horizon' not in args else args['horizon']
        self.load_weights = True if 'load-model' in args else False
        self.load_weights_path = "" if 'load-model' not in args else args['load-model']
        self.save_weights = True if 'save-model' in args else False
        self.save_model_epoch = float("inf") if 'save-model-epoch' not in args else args['save-model-epoch']
        self.save_weights_path = "" if 'save-model' not in args else args['save-model']
        
        #input network parameters
        self.vocab_size = 20 if 'vocab-size' not in args else args['vocab-size']
        self.num_agents = 12 if 'num-agents' not in args else args['num-agents']
        self.num_landmarks = 3 if 'num-landmarks' not in args else args['num-landmarks']
        # input size is just num_agents + num_landmarks
        self.hidden_comm_size = 10 if 'hidden-comm-size' not in args else args['hidden-input-size']
        self.hidden_input_size = 15 if 'hidden-input-size' not in args else args['hidden-input-size']
        self.hidden_output_size = 20 if 'hidden-output-size' not in args else args['hidden-output-size']
        self.comm_output_size = 25 if 'comm_output_size' not in args else args['comm_output_size']
        self.input_output_size = 30 if 'input_output_size' not in args else args['input_output_size']


        #hyperparameters
        self.learning_rate = 0.005 if 'learning-rate' not in args else args['learning-rate']
        self.optimizer_decay = float("-inf") if 'optimizer-decay-epoch' not in args else args['optimizer-decay']
        self.optimizer_decay_rate = 1 if 'optimizer-decay-rate' not in args else args['optimizer-decay-rate']
        self.dropout = 0 if 'dropout' not in args else args['dropout']
        self.dirichlet_alpha = 0.01 if 'dirichlet-alpha' not in args else args['dirichlet-alpha']
        self.deterministic_goals = True if 'deterministic-goals' not in args else args['deterministic-goals']
