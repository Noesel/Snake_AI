"""Pure JAX Snake rules: fixed-size state, ring-buffer body, no host callbacks."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


WIDTH = 21
CAPACITY = WIDTH * WIDTH
VECTORS = jnp.array(((0, -1), (1, 0), (0, 1), (-1, 0)), dtype=jnp.int32)


class SnakeState(NamedTuple):
    head: jax.Array
    direction: jax.Array
    body: jax.Array
    tail: jax.Array
    length: jax.Array
    occupied: jax.Array
    food: jax.Array
    energy: jax.Array
    score: jax.Array
    steps: jax.Array
    done: jax.Array
    collided: jax.Array
    key: jax.Array


def cell(position):
    return jnp.clip(position[1], 0, WIDTH - 1) * WIDTH + jnp.clip(position[0], 0, WIDTH - 1)


def sample_food(key, occupied):
    """Uniform free-cell sampling, including nearly full and full boards."""
    key, draw_key = jax.random.split(key)
    count = jnp.sum(~occupied)
    rank = jax.random.randint(draw_key, (), 0, jnp.maximum(count, 1))
    chosen = jnp.argmax(jnp.cumsum(~occupied) > rank)
    return key, jnp.where(count > 0, chosen, -1).astype(jnp.int32)


def reset(key):
    head = jnp.array((9, 9), dtype=jnp.int32)
    initial = jnp.array((7 * WIDTH + 9, 8 * WIDTH + 9, 9 * WIDTH + 9), dtype=jnp.int32)
    body = jnp.full(CAPACITY, -1, dtype=jnp.int32).at[:3].set(initial)
    occupied = jnp.zeros(CAPACITY, dtype=jnp.bool_).at[initial].set(True)
    key, food = sample_food(key, occupied)
    number = lambda value: jnp.array(value, dtype=jnp.int32)
    return SnakeState(head, number(2), body, number(0), number(3), occupied,
                      food, number(20), number(0), number(0), jnp.array(False),
                      jnp.array(False), key)


def observe(state):
    headings = (state.direction + jnp.array((0, -1, 1))) % 4
    candidates = state.head + VECTORS[headings]
    outside = jnp.any((candidates < 0) | (candidates >= WIDTH), axis=1)
    cells = jax.vmap(cell)(candidates)
    tail_cell = state.body[state.tail]
    blocked = state.occupied[cells] & (cells != tail_cell)
    food_position = jnp.array((state.food % WIDTH, state.food // WIDTH))
    delta = food_position - state.head
    forward = VECTORS[state.direction]
    right = VECTORS[(state.direction + 1) % 4]
    food_directions = jnp.sign(jnp.array((jnp.dot(delta, forward), jnp.dot(delta, right))))
    return jnp.concatenate(((outside | blocked).astype(jnp.float32),
                            food_directions.astype(jnp.float32)))


def step(state, output):
    """Advance a live episode; completed lanes stay unchanged under vmap."""
    def advance(current):
        turn = jnp.where(output > 0.5, 1, jnp.where(output < -0.5, -1, 0))
        direction = (current.direction + turn) % 4
        head = current.head + VECTORS[direction]
        next_cell = cell(head)
        old_tail = current.body[current.tail]
        outside = jnp.any((head < 0) | (head >= WIDTH))
        collided = outside | (current.occupied[next_cell] & (next_cell != old_tail))
        ate = ~collided & (next_cell == current.food)
        length = current.length + ate.astype(jnp.int32)
        occupied = current.occupied.at[old_tail].set(ate).at[next_cell].set(True)
        insertion = (current.tail + current.length) % CAPACITY
        body = current.body.at[insertion].set(next_cell)
        tail = jnp.where(ate, current.tail, (current.tail + 1) % CAPACITY)
        # Food placement happens after growth. A full board terminates safely.
        key, food = jax.lax.cond(ate & (length < CAPACITY),
                                 lambda: sample_food(current.key, occupied),
                                 lambda: (current.key, current.food))
        food = jnp.where(length == CAPACITY, -1, food)
        energy = current.energy - 1 + ate.astype(jnp.int32) * 20
        return SnakeState(head, direction, jnp.where(collided, current.body, body),
                          jnp.where(collided, current.tail, tail), length,
                          jnp.where(collided, current.occupied, occupied), food, energy,
                          current.score + ate.astype(jnp.int32), current.steps + 1,
                          collided | (energy <= 0) | (length == CAPACITY), collided, key)

    return jax.lax.cond(state.done, lambda current: current, advance, state)
