# src/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pygame
import random
from collections import deque
from src.entities.enemy import Enemy
from src.entities.bullet import Bullet
from pygame.math import Vector2


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class Trainer:
    def __init__(self, window_size=800):
        pygame.init()
        self.window_size = window_size
        self.screen = pygame.display.set_mode((window_size, window_size))
        self.clock = pygame.time.Clock()

        # 训练参数
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.train_start = 1000

        # 创建模型
        self.enemy = Enemy(window_size / 2, window_size / 2)
        self.model = DQN(self.enemy.state_size, self.enemy.action_size)
        self.optimizer = optim.Adam(self.model.parameters())

        self.bullets = []
        self.episode_rewards = []

    def generate_random_bullets(self, num_bullets=10):
        """生成随机子弹"""
        self.bullets = []
        for _ in range(num_bullets):
            pos = Vector2(random.randint(0, self.window_size),
                          random.randint(0, self.window_size))
            dir = Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
            self.bullets.append(Bullet(pos, dir))

    def get_reward(self):
        """计算奖励"""
        reward = 0.1  # 存活奖励

        # 距离中心的惩罚
        center = Vector2(self.window_size / 2, self.window_size / 2)
        dist_to_center = (self.enemy.position - center).length()
        if dist_to_center > self.window_size / 3:
            reward -= 0.1

        return reward

    def train_step(self):
        """训练一步"""
        if len(self.memory) < self.train_start:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([transition[0] for transition in batch])
        actions = torch.LongTensor([transition[1] for transition in batch])
        rewards = torch.FloatTensor([transition[2] for transition in batch])
        next_states = torch.FloatTensor([transition[3] for transition in batch])
        dones = torch.FloatTensor([transition[4] for transition in batch])

        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, episodes=1000):
        """训练模型"""
        for episode in range(episodes):
            self.enemy = Enemy(self.window_size / 2, self.window_size / 2)
            self.generate_random_bullets()
            episode_reward = 0

            for step in range(1000):
                # 处理pygame事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                # 获取当前状态
                state = self.enemy.get_state(self.bullets)

                # epsilon-greedy策略选择动作
                if random.random() < self.epsilon:
                    action = random.randrange(self.enemy.action_size)
                    self.enemy.move(self.enemy.actions[action], self.window_size)
                else:
                    action = self.enemy.act(state, self.model)

                # 更新子弹和坦克状态
                for bullet in self.bullets:
                    bullet.update(self.window_size)
                    if self.enemy.check_collision(bullet):
                        self.enemy.life -= 1
                        if self.enemy.life <= 0:
                            self.enemy.alive = False

                # 获取奖励和下一个状态
                reward = self.get_reward()
                next_state = self.enemy.get_state(self.bullets)
                done = not self.enemy.alive

                # 存储经验
                self.memory.append((state, action, reward, next_state, done))

                # 训练网络
                self.train_step()

                # 更新epsilon
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

                episode_reward += reward

                # 绘制训练过程
                self.screen.fill((255, 255, 255))
                self.enemy.draw(self.screen)
                for bullet in self.bullets:
                    bullet.draw(self.screen)

                # 显示训练信息
                font = pygame.font.Font(None, 24)
                info_text = f"Episode: {episode + 1}, Step: {step}, Reward: {episode_reward:.2f}"
                text_surface = font.render(info_text, True, (0, 0, 0))
                self.screen.blit(text_surface, (10, 10))

                pygame.display.flip()
                self.clock.tick(60)  # 控制帧率

                if done:
                    break

            self.episode_rewards.append(episode_reward)
            print(f"Episode: {episode + 1}, Reward: {episode_reward:.2f}, Epsilon: {self.epsilon:.2f}")

            # 每100个episode保存一次模型
            if (episode + 1) % 100 == 0:
                torch.save(self.model.state_dict(), f'models/model_episode_{episode + 1}.pth')

    def load_model(self, path):
        """加载训练好的模型"""
        self.model.load_state_dict(torch.load(path))
        self.model.eval()