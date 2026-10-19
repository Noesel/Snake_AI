import pygame
import math

from c_Snake import Snake
from c_Bait import Bait
from c_Playboard import Playboard


class Game:
    def __init__(self, win, game_instance, start_x, start_y):
        self.STAT_FONT = pygame.font.SysFont("comicssans", 25)
        self.win = win
        self.clock = pygame.time.Clock()
        self.offset = 20
        self.start_point_x = start_x
        self.start_point_y = start_y
        self.screen_width = 220
        self.screen_height = 220
        self.move_distance = 10
        self.end_point_x = self.start_point_x + self.screen_width
        self.end_point_y = self.start_point_y + self.screen_height

        self.playboard = Playboard(self.start_point_x, self.start_point_y, self.end_point_x, self.end_point_y, self.offset)
        self.snake = Snake(self.start_point_x, self.start_point_y, self.end_point_x, self.end_point_y, self.move_distance, self.offset)

        self.bait = Bait(self.start_point_x, self.start_point_y, self.screen_width, self.screen_height, self.move_distance, self.offset)
        self.bait.generate_pos(self.snake)
        self.got_bait = False
        self.score = 0
        self.game_instance = game_instance
        self.win.fill((50, 50, 50))

    def draw_game(self, win):
        self.playboard.draw(win)
        self.bait.draw(win)

        score_text = self.STAT_FONT.render("Score: " + str(self.score), 1, (255, 255, 255))
        win.blit(score_text, (self.end_point_x - 5 - score_text.get_width(), self.start_point_y + 3))

        game_text = self.STAT_FONT.render("Game " + str(self.game_instance + 1), 1, (255, 255, 255))
        win.blit(game_text, (self.start_point_x + 5, self.start_point_y + 3))

        self.snake.draw(win)

    def run(self):
        self.got_bait = False
        if not self.snake.collided:
            self.snake.move()

        if self.snake.x_head_pos == self.bait.x and self.snake.y_head_pos == self.bait.y:
            self.got_bait = True
            self.bait.generate_pos(self.snake)
            self.snake.grow()
            self.score += 1

        self.draw_game(self.win)

    def get_observation(self):
        """Return three danger flags and two relative food direction signs.

        Order: danger ahead/left/right, food ahead/behind, food right/left.
        Danger is 0 or 1; food is -1, 0, or 1 (positive: ahead/right).
        """
        vectors = {0: (0, -1), 90: (1, 0), 180: (0, 1), 270: (-1, 0)}
        forward_x, forward_y = vectors[self.snake.direction]
        left = (forward_y, -forward_x)
        right = (-forward_y, forward_x)
        step = self.snake.body_part_distance
        # The tail vacates its cell before Snake.move checks self-collisions.
        occupied = set(self.snake.body_parts[:-1])
        danger = []
        for dx, dy in ((forward_x, forward_y), left, right):
            next_x = self.snake.x_head_pos + dx * step
            next_y = self.snake.y_head_pos + dy * step
            hits_wall = (next_x <= self.start_point_x or next_x >= self.end_point_x
                         or next_y <= self.start_point_y + self.offset
                         or next_y >= self.end_point_y + self.offset)
            danger.append(int(hits_wall or (next_x, next_y) in occupied))

        food_x = self.bait.x - self.snake.x_head_pos
        food_y = self.bait.y - self.snake.y_head_pos
        food_forward = food_x * forward_x + food_y * forward_y
        food_right = food_x * right[0] + food_y * right[1]
        return (*danger, (food_forward > 0) - (food_forward < 0),
                (food_right > 0) - (food_right < 0))

    def calc_border_distance(self):
        """Return distances to the last safe cells ahead, left, and right."""
        distances = {
            0: max(0, self.snake.y_head_pos - (self.start_point_y + self.offset + self.move_distance)),
            90: max(0, self.end_point_x - self.move_distance - self.snake.x_head_pos),
            180: max(0, self.end_point_y + self.offset - self.move_distance - self.snake.y_head_pos),
            270: max(0, self.snake.x_head_pos - (self.start_point_x + self.move_distance)),
        }
        direction = self.snake.direction
        return (distances[direction], distances[(direction - 90) % 360],
                distances[(direction + 90) % 360])

    def distance_to_bait(self, bait_x, bait_y, snake_x, snake_y):
        erg = math.sqrt(math.pow(bait_x - snake_x, 2) + math.pow(bait_y - snake_y, 2))
        return erg