import random

class Bait:
    def __init__(self, start_point_x, start_point_y, screen_width, screen_height, move_distance, offset, rng=None):
        self.rng = rng if rng is not None else random
        self.x = 0
        self.y = 0
        self.screen_startpoint_x = start_point_x
        self.screen_startpoint_y = start_point_y
        self.offset = offset
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.circle_radius = int(move_distance / 2)

    def draw(self, win):
        import pygame

        if self.x is None:
            return
        pygame.draw.circle(win, (0, 255, 0), (self.x, self.y), self.circle_radius)

    def generate_pos(self, snake):
        occupied = set(snake.body_parts)
        # Rejection sampling is fast on sparse boards; bound retries on dense ones.
        for _ in range(64):
            x = self.rng.randint(1, self.screen_width // 10 - 1) * 10 + self.screen_startpoint_x
            y = self.rng.randint(1, self.screen_height // 10 - 1) * 10 + self.screen_startpoint_y + self.offset
            if (x, y) not in occupied:
                self.x, self.y = x, y
                return True

        free = [(self.screen_startpoint_x + x, self.screen_startpoint_y + self.offset + y)
                for y in range(10, self.screen_height, 10)
                for x in range(10, self.screen_width, 10)
                if (self.screen_startpoint_x + x, self.screen_startpoint_y + self.offset + y) not in occupied]
        if not free:
            self.x = self.y = None
            return False
        self.x, self.y = self.rng.choice(free)
        return True
