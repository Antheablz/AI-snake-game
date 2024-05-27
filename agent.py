import torch
import random
import numpy as np

from collections import deque
from snake_game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from graph import plot

MAX_MEM = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001


class Agent:
    def __init__(self):
        self.num_games = 0
        self.epsilon = 0  # parameter to control the randomness
        self.gamma = 0.9  # discount rate, must be < 1
        self.memory = deque(
            maxlen=MAX_MEM
        )  # if we exceed max_mem then automatically pop elements w/ popleft()
        self.model = Linear_QNet(11, 256, 3)  # need 11 inputs and 3 outputs.
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r))
            or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d)),
            # Danger right
            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u))
            or (dir_r and game.is_collision(point_d)),
            # Danger left
            (dir_d and game.is_collision(point_r))
            or (dir_u and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u))
            or (dir_l and game.is_collision(point_d)),
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
        ]

        return np.array(state, dtype=int)

    def get_action(self, state):  # gets action based on state
        # random moves: tradeoff exploration / exploitation
        # so make random moves to explore the environment but the better out agent gets, the less random moves it makes
        # and the more we want to exploit our agent
        self.epsilon = (
            80 - self.num_games
        )  # The more games we have the smaller our epsilon is
        next_move = [0, 0, 0]
        if (
            random.randint(0, 200) < self.epsilon
        ):  # the smaller the epsilon, the less frequent we choose a random move
            move = random.randint(0, 2)  # give random move- 0, 1, or 2
            next_move[move] = 1
        else:  # make a move based on a our model,
            state0 = torch.tensor(state, dtype=torch.float)  # convert state to a tensor
            prediction = self.model(
                state0
            )  # get our prediction based on a single state
            move = torch.argmax(
                prediction
            ).item()  # return type is a tesnor so we convert to single int by calling item()
            next_move[move] = 1

        return next_move

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append(
            (state, action, reward, next_state, game_over)
        )  # if exceeds max_mem then popleft(), append as one element

    def train_long_memory(self):
        # grep 1000 (a single batch) from memory
        # get the batch_size number of tuples
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(
                self.memory, BATCH_SIZE
            )  # returns a list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        # for only one step
        self.trainer.train_step(state, action, reward, next_state, game_over)


def train():  # global function
    plot_scores = []  # empty list at start to keep track of scores
    plot_avg_scores = []
    total_score = 0
    cur_record = 0

    # create an agent and a game
    agent = Agent()
    game = SnakeGameAI()

    # create training loop, runs until we quit script
    while True:
        # get old state
        cur_state = agent.get_state(game)

        # get move based on current state
        next_move = agent.get_action(cur_state)

        # preform the move and get new state
        reward, game_over, score = game.play_step(next_move)
        next_state = agent.get_state(game)

        # now train agents short memory (which is only for a single step)
        agent.train_short_memory(cur_state, next_move, reward, next_state, game_over)

        # remember all of the above and store in long term memory (which is a deque)
        agent.remember(cur_state, next_move, reward, next_state, game_over)

        if game_over:
            # train the long remory (or replay memory), trains on all previous moves and games our agent has played, also plot result
            game.reset()
            agent.num_games += 1
            agent.train_long_memory()

            if score > cur_record:
                cur_record = score
                agent.model.save()

            print(
                f"Total Games: {agent.num_games}  Score: {score} Record: {cur_record}"
            )

            # after each game must append scores
            plot_scores.append(score)
            total_score += score
            avg_score = total_score / agent.num_games
            plot_avg_scores.append(avg_score)
            plot(plot_scores, plot_avg_scores)


if __name__ == "__main__":
    train()
