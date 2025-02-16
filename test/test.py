# src/game/game.py
import pygame
import sys
from pygame.math import Vector2
from typing import List, Optional


class Bullet:
    """子弹类"""

    def __init__(self, position: Vector2, direction: Vector2, speed: float = 7):
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

    def update(self, screen_size: int) -> None:
        """更新子弹位置和状态"""
        # 更新位置
        self.position += self.direction * self.speed
        self.rect.center = self.position

        # 边界碰撞检测
        if (self.position.x - self.radius < 0 or
                self.position.x + self.radius > screen_size or
                self.position.y - self.radius < 0 or
                self.position.y + self.radius > screen_size):

            if self.bounced:
                self.active = False
            else:
                self.direction = Vector2(
                    -self.direction.x if self.position.x - self.radius < 0 or
                                         self.position.x + self.radius > screen_size else self.direction.x,
                    -self.direction.y if self.position.y - self.radius < 0 or
                                         self.position.y + self.radius > screen_size else self.direction.y
                )
                self.bounced = True

    def draw(self, screen: pygame.Surface) -> None:
        """绘制子弹"""
        pygame.draw.circle(screen, (0, 0, 0), self.position, self.radius)


class Tank:
    """坦克基类"""

    def __init__(self, x: float, y: float, size: int = 20, speed: float = 2):
        self.position = Vector2(x, y)
        self.size = size
        self.speed = speed
        self.direction = Vector2(0, -1)
        self.alive = True
        self.shoot_cooldown = 30
        self.cooldown_counter = 0
        self.rect = pygame.Rect(x - size / 2, y - size / 2, size, size)

    def move(self, direction: Vector2, screen_size: int) -> None:
        """移动坦克"""
        if not self.alive or direction.length() == 0:
            return

        # 计算新位置
        new_position = self.position + direction.normalize() * self.speed

        # 边界检查
        half_size = self.size / 2
        if (new_position.x - half_size < 0 or
                new_position.x + half_size > screen_size or
                new_position.y - half_size < 0 or
                new_position.y + half_size > screen_size):
            return

        # 更新位置和方向
        self.position = new_position
        self.direction = direction.normalize()
        self.rect.center = self.position

    def shoot(self) -> Optional[Bullet]:
        """发射子弹"""
        if not self.alive or self.cooldown_counter > 0:
            return None

        self.cooldown_counter = self.shoot_cooldown
        return Bullet(self.position + self.direction * self.size, self.direction)

    def update(self, screen_size: int) -> Optional[Bullet]:
        """更新状态"""
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        return None

    def check_collision(self, bullet: Bullet) -> bool:
        """检查与子弹的碰撞"""
        return self.alive and self.rect.colliderect(bullet.rect)

    def draw(self, screen: pygame.Surface) -> None:
        """绘制坦克"""
        if self.alive:
            pygame.draw.rect(screen, (0, 0, 0), self.rect)
            # 绘制炮管方向
            end_pos = self.position + self.direction * self.size
            pygame.draw.line(screen, (0, 0, 0), self.position, end_pos, 3)


class Player(Tank):
    """玩家控制的坦克"""

    def update(self, screen_size: int) -> Optional[Bullet]:
        """处理玩家输入并更新状态"""
        super().update(screen_size)

        if not self.alive:
            return None

        # 处理移动输入
        keys = pygame.key.get_pressed()
        direction = Vector2(0, 0)
        if keys[pygame.K_w]: direction.y -= 1
        if keys[pygame.K_s]: direction.y += 1
        if keys[pygame.K_a]: direction.x -= 1
        if keys[pygame.K_d]: direction.x += 1

        if direction.length() > 0:
            self.move(direction, screen_size)

        # 处理射击
        return self.shoot() if keys[pygame.K_SPACE] else None


class Game:
    """游戏主类"""

    def __init__(self, window_size: int = 800):
        pygame.init()
        self.WINDOW_SIZE = window_size
        self.screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("Tank Game")

        self.clock = pygame.time.Clock()
        self.FPS = 60
        self.reset_game()

    def reset_game(self) -> None:
        """重置游戏状态"""
        self.player = Player(self.WINDOW_SIZE / 2, self.WINDOW_SIZE - 100)
        self.enemy = Tank(self.WINDOW_SIZE / 2, 100)
        self.bullets: List[Bullet] = []
        self.game_over = False

    def update(self) -> None:
        """更新游戏状态"""
        if self.game_over:
            return

        # 更新玩家和敌人
        for tank in [self.player, self.enemy]:
            bullet = tank.update(self.WINDOW_SIZE)
            if bullet:
                self.bullets.append(bullet)

        # 更新子弹和碰撞检测
        for bullet in self.bullets[:]:
            bullet.update(self.WINDOW_SIZE)

            for tank in [self.player, self.enemy]:
                if tank.check_collision(bullet):
                    tank.alive = False
                    self.game_over = True

            if not bullet.active:
                self.bullets.remove(bullet)

    def draw(self) -> None:
        """绘制游戏画面"""
        self.screen.fill((255, 255, 255))

        # 绘制坦克和子弹
        for tank in [self.player, self.enemy]:
            tank.draw(self.screen)
        for bullet in self.bullets:
            bullet.draw(self.screen)

        # 游戏结束显示
        if self.game_over:
            font = pygame.font.Font(None, 74)
            text = font.render('Game Over', True, (0, 0, 0))
            text_rect = text.get_rect(center=(self.WINDOW_SIZE / 2, self.WINDOW_SIZE / 2))
            self.screen.blit(text, text_rect)

        pygame.display.flip()

    def run(self) -> None:
        """运行游戏主循环"""
        running = True
        while running:
            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r and self.game_over:
                        self.reset_game()

            self.update()
            self.draw()
            self.clock.tick(self.FPS)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = Game()
    game.run()