import random
import pygame
from pygame.math import Vector2


class Bullet:
    def __init__(self, position, direction, speed=7):
        """
        参数初始化
        position: 初始位置
        direction: 移动方向
        speed: 移动速度
        radius: 子弹半径
        active: 子弹状态
        bounced: 反弹状态
        rect: 碰撞矩形
        """
        self.position = Vector2(position)
        self.direction = Vector2(direction).normalize()
        self.speed = speed
        self.radius = 4
        self.active = True
        self.bounced = False
        self.rect = pygame.Rect(
            position.x - self.radius,
            position.y - self.radius,
            self.radius * 2,
            self.radius * 2
        )

    def update(self, screen_size):
        """更新子弹位置和反弹状态"""
        self.position += self.direction * self.speed
        self.rect.center = self.position

        hit_horizontal = self.position.x - self.radius < 0 or self.position.x + self.radius > screen_size
        hit_vertical = self.position.y - self.radius < 0 or self.position.y + self.radius > screen_size

        if hit_horizontal or hit_vertical:
            if self.bounced:
                self.active = False
            else:
                self.direction = Vector2(
                    -self.direction.x if hit_horizontal else self.direction.x,
                    -self.direction.y if hit_vertical else self.direction.y
                )
                if random.random() < 0.3:
                    self.bounced = True

    def draw(self, screen):
        """绘制子弹"""
        pygame.draw.circle(screen, (0, 0, 0), self.position, self.radius)
