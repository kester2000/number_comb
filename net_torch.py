import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Net(nn.Module):
    """policy-value network module"""

    def __init__(self):
        super(Net, self).__init__()

        # common layers
        self.fc1 = nn.Linear(63, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)

        # action policy layers
        self.act_fc1 = nn.Linear(512, 64)
        self.act_fc2 = nn.Linear(64, 20)
        # state value layers
        self.val_fc1 = nn.Linear(512, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = torch.relu(self.fc1(state_input))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        # action policy layers
        x_act = torch.relu(self.act_fc1(x))
        x_act = torch.log_softmax(self.act_fc2(x_act), dim=0)
        # state value layers
        x_val = torch.relu(self.val_fc1(x))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyValueNet():
    """policy-value network """

    def __init__(self, model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = Net().cuda()
        else:
            self.policy_value_net = Net()
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def policy_value_fn(self, board, card):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_actions = board.get_actions()
        current_state = np.ascontiguousarray(board.get_state(card))
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_value_net(
                Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_actions, act_probs[legal_actions])
        value = value.data[0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, values, lr):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            values = Variable(torch.FloatTensor(values).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            values = Variable(torch.FloatTensor(values))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), values)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
            torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
        )
        return loss.item(), entropy.item()
        # for pytorch version >= 0.5 please use the following line instead.
        # return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)
