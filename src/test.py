import time

import pygame
import torch
import os
import random
from pygame.math import Vector2
from src.entities.bullet import Bullet
from src.entities.enemy import Enemy
from src.trainer import DQN


class TestingEnvironment:
    def __init__(self, window_size=800):
        self.start = None
        self.model = None
        pygame.init()
        self.WINDOW_SIZE = window_size
        self.screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("Testing AI Tank")
        self.clock = pygame.time.Clock()
        self.FPS = 60

        self.enemy = Enemy(window_size / 2, window_size / 2)
        self.bullets = []
        self.generate_random_bullets()

    def load_latest_model(self, path):
        try:
            self.model = DQN(state_size=42, action_size=5)
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")

    def generate_random_bullets(self, num_bullets=10):
        """生成随机子弹"""
        self.bullets = []
        for _ in range(num_bullets):
            pos = Vector2(random.randint(0, self.WINDOW_SIZE),
                          random.randint(0, self.WINDOW_SIZE))
            dir = Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
            self.bullets.append(Bullet(pos, dir))

    def reset_bullets(self):
        """重置子弹"""
        self.generate_random_bullets()

    def run(self):
        running = True
        flag_time = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            if flag_time:
                self.start = time.time()
                flag_time = False
            bullet = self.enemy.update(self.bullets, self.model)
            if bullet and isinstance(bullet, list):  # 检查返回值是否为列表
                for bulls in bullet:
                    self.bullets.append(bulls)

            # 更新子弹位置
            for bullet in self.bullets[:]:
                bullet.update(self.WINDOW_SIZE)
                # 更新后继续检测是否有碰撞
                for tank in [self.enemy]:
                    if tank.check_collision(bullet):
                        tank.life -= 1
                        bullet.active = False
                        if not bullet.active:
                            self.bullets.remove(bullet)
                        if tank.life == 0:
                            print("AI Tank is destroyed!")
                            self.enemy.life = 3
                            self.enemy.alive = True
                            self.reset_bullets()
                            flag_time = True

            self.screen.fill((255, 255, 255))
            self.enemy.draw(self.screen)
            for bullet in self.bullets:
                bullet.draw(self.screen)

            # 显示信息
            font = pygame.font.Font(None, 36)
            text = font.render(f"Life: {self.enemy.life}  time: {int(time.time() - self.start)}", True, (0, 0, 0))
            self.screen.blit(text, (10, 10))

            pygame.display.flip()
            self.clock.tick(self.FPS)

        pygame.quit()
