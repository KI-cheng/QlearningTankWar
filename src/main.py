# src/main.py
import argparse
from src.game.game import Game
from src.trainer import Trainer
import os

if __name__ == "__main__":
    flag = "play"

    if flag == 'train':
        trainer = Trainer(window_size=800)
        trainer.train(episodes=1000)
    elif flag == 'play':
        model_path = "models/first_version.pth"

        game = Game()
        game.load_ai_model(model_path)
        game.run()