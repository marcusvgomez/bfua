"""
Training config

Contains runtime config params
"""

class RuntimeConfig:

    def __init__(self, args):
        self.using_cuda = args['use-cuda']
        self.n_epochs = 10000 if 'n-epochs' not in args else args['n-epochs']
        self.time_horizon = 1000 if 'horizon' not in args else args['horizon']
        self.load_weights = True if 'load-model' in args else False
        self.load_weights_path = "" if 'load-model' not in args else args['load-model']
        self.save_weights = True if 'save-model' in args else False
        self.save_weights_path = "" if 'save-model' not in args else args['save-model']
        self.vocab_size = 20 if 'vocab-size' not in args else args['vocab-size']

