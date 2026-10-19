"""Optional pygame visualization. Workers never import this module."""

import random

from c_Game import Game
from evaluation import EpisodeResult, GenomeResult, step_game


class PopulationRenderer:
    def __init__(self, fps=60):
        import pygame

        self.pygame = pygame
        pygame.init()
        self.window = pygame.display.set_mode((1850, 1000))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 25)
        self.fps = fps
        self.generation = 0

    def close(self):
        self.pygame.quit()

    def evaluate(self, genomes, config, seeds, max_steps):
        import neat

        pygame = self.pygame
        episodes = {gid: [] for gid, _ in genomes}
        for start in range(0, len(genomes), 24):
            batch = genomes[start:start + 24]
            networks = {gid: neat.nn.FeedForwardNetwork.create(genome, config)
                        for gid, genome in batch}
            for episode_number, seed in enumerate(seeds, 1):
                self.window.fill((50, 50, 50))
                active = []
                for index, (gid, _) in enumerate(batch):
                    game = Game(self.window, start + index, (index % 8) * 230,
                                20 + (index // 8) * 280, rng=random.Random(seed))
                    active.append([gid, game, 0])
                while active:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt
                    remaining = []
                    for gid, game, steps in active:
                        step_game(game, networks[gid])
                        steps += 1
                        if game.done or steps >= max_steps:
                            episodes[gid].append(EpisodeResult(game.score, steps))
                        else:
                            remaining.append([gid, game, steps])
                    active = remaining
                    pygame.draw.rect(self.window, (50, 50, 50), (0, 0, 1850, 20))
                    text = f"Generation {self.generation + 1} | Episode {episode_number}/{len(seeds)} | Genomes {start + 1}-{start + len(batch)}"
                    self.window.blit(self.font.render(text, True, (255, 255, 255)), (10, 0))
                    pygame.display.flip()
                    if self.fps:
                        self.clock.tick(self.fps)
        self.generation += 1
        return {gid: GenomeResult(tuple(results)) for gid, results in episodes.items()}
