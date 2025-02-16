# src/entities/enemy.py
import torch
import numpy as np
from pygame.math import Vector2
from .tank import Tank


class Enemy(Tank):
    def __init__(self, x, y, size=20, speed=5):
        super().__init__(x, y, size, speed)
        self.state_size = 8  # 状态空间维度
        self.action_size = 5  # 动作空间维度(上下左右停)

        # 动作映射表
        self.actions = {
            0: Vector2(0, -1),  # 上
            1: Vector2(0, 1),  # 下
            2: Vector2(-1, 0),  # 左
            3: Vector2(1, 0),  # 右
            4: Vector2(0, 0)  # 停
        }

    def get_state(self, bullets):
        """获取状态向量"""
        state = []
        # 坦克位置的归一化坐标
        state.extend([self.position.x / 800, self.position.y / 800])

        # 找最近的3个子弹
        bullet_info = []
        for bullet in bullets[:3]:
            # 子弹相对位置
            rel_pos = bullet.position - self.position
            bullet_info.extend([rel_pos.x / 800, rel_pos.y / 800])

        # 如果子弹少于3个，用0填充
        while len(bullet_info) < 6:
            bullet_info.extend([0, 0])

        state.extend(bullet_info)
        return np.array(state)

    def act(self, state, model):
        """根据状态选择动作"""
        if not self.alive:
            return None

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor)
            action = q_values.max(1)[1].item()

        # 移动坦克
        self.move(self.actions[action], 800)
        return action

    def update(self, bullets, model=None):
        """更新敌方坦克状态"""
        super().update(800)

        if model is None or not self.alive:
            return None

        state = self.get_state(bullets)
        return self.act(state, model)