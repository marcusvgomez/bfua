from models.agent import * 
from models.controller import *
from models.env import *
import argparse
import config


def main():
    parser = argparse.ArgumentParser(description="Train time babbbyyyyyyyy")
    parser.add_argument('--n-epochs', 'e', type=int, help="Optional param specifying the number of epochs")
    parser.add_argument('--horizon', 't', type=int, help="Optional param specifying the number of timesteps in a given epoch")
    parser.add_argument('--use-cuda', action='store_true', help="Optional param that specifies whether to train on cuda")
    parser.add_argument('--load-model', type=str, help='Optional param that specifies model weights to start using')
    parser.add_argument('--save-model', type=str, help='Optional param that specifies where to save model weights')
    parser.add_argument('--vocab-size', type=int, help='Optinoal param that specifies maximum vocabulary size')
    args = vars(parser.parse_args())
    runtime_config = RuntimeConfig(args)
    controller = Controller()
    optimizer = RMSprop(controller.agent.parameters(), lr=0.01)
    for epoch in range(runtime_config.n_epochs):
        controller.reset()
        for t in range():
            controller.step()
            optimizer.zero_grad()
            total_loss = controller.compute_loss()
            total_loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()

