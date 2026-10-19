"""Optional preview of the best TensorNEAT genome; training stays on device."""

import jax
import jax.numpy as jnp
import numpy as np

import jax_snake as snake
from c_Game import Game


class TensorRenderer:
    def __init__(self, algorithm, fps=60):
        import pygame
        self.pygame = pygame
        pygame.init()
        self.window = pygame.display.set_mode((520, 600))
        self.canvas = pygame.Surface((260, 300))
        self.game = Game(self.canvas, 0, 20, 20)
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.algorithm = algorithm
        self.advance = jax.jit(lambda state, params, game: snake.step(
            game, algorithm.forward(state, params, snake.observe(game))[0]))

    def close(self):
        self.pygame.quit()

    def show(self, state, best, config, generation):
        pygame = self.pygame
        pygame.display.set_caption(f"TensorNEAT generation {generation} | best fitness {best['fitness']:.1f}")
        params = self.algorithm.transform(state, (jnp.asarray(best['nodes']), jnp.asarray(best['conns'])))
        game = snake.reset(state.snake_episode_keys[0])
        for _ in range(config.max_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
            game = self.advance(state, params, game)
            snapshot = jax.device_get(game)
            indices = (int(snapshot.tail) + np.arange(int(snapshot.length))) % snake.CAPACITY
            body = snapshot.body[indices][::-1]
            self.game.snake.body_parts = [(20 + 10 + int(cell % snake.WIDTH) * 10,
                                           20 + 20 + 10 + int(cell // snake.WIDTH) * 10) for cell in body]
            food = int(snapshot.food)
            self.game.bait.x = None if food < 0 else 20 + 10 + (food % snake.WIDTH) * 10
            self.game.bait.y = None if food < 0 else 20 + 20 + 10 + (food // snake.WIDTH) * 10
            self.game.score = int(snapshot.score)
            self.canvas.fill((50, 50, 50))
            self.game.draw_game(self.canvas)
            self.window.blit(pygame.transform.scale(self.canvas, self.window.get_size()), (0, 0))
            pygame.display.flip()
            if self.fps:
                self.clock.tick(self.fps)
            if bool(snapshot.done):
                break
