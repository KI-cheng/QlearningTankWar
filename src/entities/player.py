from pygame.math import Vector2
import pygame
from .tank import Tank


class Player(Tank):
    """玩家控制的坦克类"""

    def update(self, screen_size):
        """处理玩家输入并更新状态"""
        super().update(screen_size)

        if not self.alive:
            return None

        # 处理移动输入
        keys = pygame.key.get_pressed()
        direction = Vector2(0, 0)
        if keys[pygame.K_w]:
            direction.y -= 1  # 向上移动
        if keys[pygame.K_s]:
            direction.y += 1  # 向下移动
        if keys[pygame.K_a]:
            direction.x -= 1  # 向左移动
        if keys[pygame.K_d]:
            direction.x += 1  # 向右移动

        if direction.length() > 0:
            self.move(direction, screen_size)

        # 处理射击
        return self.shoot() if keys[pygame.K_SPACE] else None
