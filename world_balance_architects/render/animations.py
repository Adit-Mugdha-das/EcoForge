"""
Animation helpers for EcoForge renderer effects.

This module keeps reusable visual systems separate from renderer.py,
so the renderer can focus on composition and layout.
"""

from __future__ import annotations

import math
import random

import pygame


class WaterAnimator:
    """Alternates between two water frames every N ticks."""

    def __init__(self, fps_interval: int = 28):
        self._tick = 0
        self._interval = fps_interval
        self._frame = 0

    def update(self) -> None:
        self._tick += 1
        if self._tick >= self._interval:
            self._tick = 0
            self._frame = 1 - self._frame

    @property
    def frame(self) -> int:
        return self._frame


class FloatingText:
    """A text label that rises and fades over a fixed lifetime."""

    LIFETIME = 52

    def __init__(
        self,
        text: str,
        world_col: int,
        world_row: int,
        tile_size: int,
        color: tuple[int, int, int] = (255, 255, 200),
    ):
        self.text = text
        self.color = color
        self.age = 0

        self.x = float(world_col * tile_size + tile_size // 2 + random.randint(-6, 6))
        self.y = float(world_row * tile_size + tile_size // 2)

    def update(self) -> None:
        self.age += 1
        self.y -= 1.4

    @property
    def alive(self) -> bool:
        return self.age < self.LIFETIME

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        if not self.alive:
            return
        alpha = int(255 * max(0, 1 - self.age / self.LIFETIME))
        surface = font.render(self.text, True, self.color)
        surface.set_alpha(alpha)
        screen.blit(surface, surface.get_rect(center=(int(self.x), int(self.y))))


class Particle:
    """A small particle that bursts outward and fades."""

    def __init__(self, pos: tuple[float, float], color: tuple[int, int, int]):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1.8, 4.0)
        self.x, self.y = float(pos[0]), float(pos[1])
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed - 1.8
        self.color = color
        self.age = 0
        self.life = random.randint(28, 42)

    def update(self) -> None:
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.12
        self.age += 1

    @property
    def alive(self) -> bool:
        return self.age < self.life

    def draw(self, screen: pygame.Surface) -> None:
        if not self.alive:
            return

        alpha = int(255 * (1 - self.age / self.life))
        radius = max(1, 4 - self.age // 10)

        surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(surface, (*self.color, alpha), (radius, radius), radius)
        screen.blit(surface, (int(self.x) - radius, int(self.y) - radius))


class CameraShake:
    """Short-lived screen shake offset generator."""

    def __init__(self):
        self._amount = 0
        self._frames = 0

    def trigger(self, amount: int = 5, frames: int = 18) -> None:
        self._amount = amount
        self._frames = frames

    def offset(self) -> tuple[int, int]:
        if self._frames <= 0:
            return (0, 0)

        self._frames -= 1
        amount = self._amount
        return (random.randint(-amount, amount), random.randint(-amount, amount))


__all__ = [
    "WaterAnimator",
    "FloatingText",
    "Particle",
    "CameraShake",
]
