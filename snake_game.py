"""
modified from the original from https://github.com/patrickloeber/python-fun/tree/master/snake-pygame
"""

import pygame
import random
import numpy as np

from collections import namedtuple
from enum import Enum


pygame.init()
font = pygame.font.Font("arial.ttf", 25)
# font = pygame.font.SysFont('arial', 25)


# RESET: after each game agent should be able to reset game and start new game
# REWAWARD that our agent gets
# PLAY(ACTION) -> computes direction
# keep track of game_iteration
# change is_collision


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20  # how big block size is in pixels
SPEED = 10  # 20


class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        # self.direction = list(direction.values()).index(random.randint(1,4))
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE

        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            # if event.type == pygame.KEYDOWN:
            # if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT or event.key == pygame.K_UP or event.key == pygame.K_DOWN:
            # chosen = direction[pygame.key.name(event.key)]
            # elf.direction = chosen

        # 2. move
        self._move(action)
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, point=None):
        if point == None:
            point = self.head
        # hits boundary
        if (
            point.x > self.w - BLOCK_SIZE
            or point.x < 0
            or point.y > self.h - BLOCK_SIZE
            or point.y < 0
        ):
            return True
        # hits itself
        if point in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for point in self.snake:
            pygame.draw.rect(
                self.display,
                BLUE1,
                pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE),
            )
            pygame.draw.rect(
                self.display, BLUE2, pygame.Rect(point.x + 4, point.y + 4, 12, 12)
            )

        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE),
        )

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]

        step_clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        cur_index = step_clockwise.index(self.direction)

        if np.array_equal(
            action, [1, 0, 0]
        ):  # if want to go straight, no direction change
            new_step = step_clockwise[cur_index]

        if np.array_equal(
            action, [0, 1, 0]
        ):  # if want to turn right, change direction in clockwise order
            new_step = step_clockwise[(cur_index + 1) % 4]

        if np.array_equal(
            action, [0, 0, 1]
        ):  # if want to turn left, change direction in anti-clockwise order
            new_step = step_clockwise[(cur_index - 1) % 4]

        self.direction = new_step

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)


if __name__ == "__main__":
    game = SnakeGameAI()

    # game loop
    while True:
        game_over, score = game.play_step()

        if game_over == True:
            break

    print("Final Score", score)

    pygame.quit()
