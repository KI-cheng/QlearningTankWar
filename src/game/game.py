import pygame
import sys

from src.entities.enemy import Enemy
from src.entities.player import Player
from src.entities.tank import Tank
from src.trainer import DQN  # 导入DQN模型
import torch


class Game:
    def __init__(self, window_size=800):
        """
        window_size: 窗口尺寸
        screen: 游戏屏幕
        clock: 游戏时钟
        FPS: 游戏帧率
        player: 玩家坦克
        enemy: 敌方坦克
        bullets: 子弹列表
        game_over: 游戏状态
        """
        self.game_over = None
        self.bullets = None
        self.enemy = None
        self.player = None
        self.model = None  # 添加模型属性
        pygame.init()
        self.WINDOW_SIZE = window_size
        self.screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("Tank War")

        self.clock = pygame.time.Clock()
        self.FPS = 60
        self.reset_game()

    def load_ai_model(self, model_path):
        """加载AI模型"""
        try:
            self.model = DQN(state_size=42, action_size=5)
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    def reset_game(self):
        """重置游戏状态"""
        self.player = Player(self.WINDOW_SIZE / 2, self.WINDOW_SIZE - 100)
        self.enemy = Enemy(self.WINDOW_SIZE / 2, 100)
        self.bullets = []
        self.game_over = False

    def update(self):
        """更新游戏状态和碰撞检测"""
        if self.game_over:
            return

        # 玩家：通过player更新玩家操作，然后再通过tank更新位置获取shoot返回的子弹
        bullet = self.player.update(self.WINDOW_SIZE)
        if bullet and isinstance(bullet, list):  # 检查返回值是否为列表
            for bulls in bullet:
                self.bullets.append(bulls)

        # AI敌人：通过enemy更新模型操作，继承的tank更新位置和shoot返回的子弹
        bullet = self.enemy.update(self.bullets, self.model)
        if bullet and isinstance(bullet, list):  # 检查返回值是否为列表
            for bulls in bullet:
                self.bullets.append(bulls)

        # 更新子弹位置
        for bullet in self.bullets[:]:
            bullet.update(self.WINDOW_SIZE)
            # 更新后继续检测是否有碰撞
            for tank in [self.player, self.enemy]:
                if tank.check_collision(bullet):
                    tank.life -= 1
                    bullet.active = False
                    if tank.life == 0:
                        tank.alive = False
                        self.game_over = True
            # 更新后的子弹检查是否失效，失效移除
            if not bullet.active:
                self.bullets.remove(bullet)

    def draw(self):
        """绘制游戏画面"""
        self.screen.fill((255, 255, 255))

        for tank in [self.player, self.enemy]:
            tank.draw(self.screen)
        for bullet in self.bullets:
            bullet.draw(self.screen)

        if self.game_over:
            font = pygame.font.Font(None, 74)
            text = font.render('Game Over', True, (0, 0, 0))
            text_rect = text.get_rect(center=(self.WINDOW_SIZE / 2, self.WINDOW_SIZE / 2))
            self.screen.blit(text, text_rect)

        pygame.display.flip()

    def run(self):
        """运行游戏主循环"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r and self.game_over:
                        self.reset_game()

            self.update()  # 更新游戏objects的状态
            self.draw()  # 渲染更新后的状态
            self.clock.tick(self.FPS)

        pygame.quit()
        sys.exit()