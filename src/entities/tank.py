import pygame
import random
from pygame.math import Vector2
from .bullet import Bullet


class Tank:
    def __init__(self, x, y, size=20, speed=5):
        """
        参数初始化
        position: 坦克位置
        size: 坦克尺寸
        speed: 移动速度
        direction: 朝向
        alive: 存活状态
        shoot_cooldown: 射击冷却时间
        cooldown_counter: 当前冷却计数
        rect: 碰撞矩形
        life: 生命值
        """
        self.position = Vector2(x, y)
        self.size = size
        self.speed = speed
        self.direction = Vector2(0, -1)
        self.alive = True
        self.life = 3
        self.shoot_cooldown = 30
        self.cooldown_counter = 0
        self.rect = pygame.Rect(x - size / 2, y - size / 2, size, size)

    def move(self, direction, screen_size):
        """移动坦克并处理边界碰撞"""
        if not self.alive or direction.length() == 0:
            return

        new_position = self.position + direction.normalize() * self.speed

        half_size = self.size / 2
        if (new_position.x - half_size < 0 or
                new_position.x + half_size > screen_size or
                new_position.y - half_size < 0 or
                new_position.y + half_size > screen_size):
            return

        self.position = new_position
        self.direction = direction.normalize()
        self.rect.center = self.position

    def shoot(self):
        """发射子弹并重置冷却"""
        if not self.alive or self.cooldown_counter > 0:
            return None

        self.cooldown_counter = self.shoot_cooldown

        Bullets = []
        # Bullets.append(Bullet(self.position + self.direction * self.size, self.direction))
        for i in range(1, 4):
            random_x = random.uniform(-0.1, 0.1)
            random_y = random.uniform(-0.1, 0.1)
            # 偏移量
            v = Vector2(random_x, random_y)
            Bullets.append(Bullet(self.position + (self.direction + v) * self.size, self.direction + v*3))
        # print(Bullets)
        return Bullets

    def update(self, screen_size):
        """更新冷却状态"""
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        return None

    def check_collision(self, bullet):
        """检测子弹碰撞"""
        return self.alive and self.rect.colliderect(bullet.rect)

    def draw(self, screen):
        """绘制坦克主体和方向指示器（三角形）"""
        if self.alive:
            color = {3: 0, 2: 96, 1: 192, 0: 255}
            c = (color[self.life], color[self.life], color[self.life])
            pygame.draw.rect(screen, c, self.rect)

            # 三角形三个顶点
            triangle_size = self.size * 0.5  # 三角形大小
            tip = self.position + self.direction * 2 * triangle_size  # 三角形尖端
            # 计算垂直于方向的向量用于确定三角形底边两端
            perpendicular = Vector2(-self.direction.y, self.direction.x) * (self.size * 0.4)
            base_1 = self.position + perpendicular
            base_2 = self.position - perpendicular
            # 绘制三角形
            pygame.draw.polygon(screen, c, [
                (tip.x, tip.y),
                (base_1.x, base_1.y),
                (base_2.x, base_2.y)
            ])
