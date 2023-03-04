import random
import numpy as np
from collections import deque
from board import *
from mcts import MCTSPlayer
from net_torch import PolicyValueNet


class TrainPipeline():
    def __init__(self, init_model=None):
        # params of the board and the game
        # self.board_width = 6
        # self.board_height = 6
        # self.n_in_row = 4
        # self.board = Board(width=self.board_width,
        #                    height=self.board_height,
        #                    n_in_row=self.n_in_row)
        # self.game = Game(self.board)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 100  # num of simulations for each move
        self.c_puct = 1000
        self.buffer_size = 10000
        self.batch_size = 20  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 5
        self.game_batch_num = 1500
        self.best_score = 0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet()
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            board = Board()
            cards = Board.get_random_cards()
            states = []
            probses = []
            for card in cards:
                action, probs = self.mcts_player.get_action(
                    board, card, self.temp, retrun_probs=1)
                states.append(board.get_state(card))
                probses.append(probs)
                board.do_move(card, action)
            score = board.get_score()
            play_data = list(zip(states, probses, [score]*20))[:]
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        score_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                score_batch,
                self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(score_batch) - old_v.flatten()) /
                             np.var(np.array(score_batch)))
        explained_var_new = (1 -
                             np.var(np.array(score_batch) - new_v.flatten()) /
                             np.var(np.array(score_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        sum_score = 0
        for i in range(n_games):
            board = Board()
            cards = Board.get_random_cards()
            for card in cards:
                action = current_mcts_player.get_action(board, card)
                board.do_move(card, action)
        sum_score += board.get_score()
        print(f'avg score {sum_score/n_games}')
        return sum_score/n_games

    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}".format(i+1))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    score = self.policy_evaluate()
                    self.policy_value_net.save_model('./current_policy.model')
                    if score > self.best_score:
                        print("New best policy!!!!!!!!")
                        self.best_score = score
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()
