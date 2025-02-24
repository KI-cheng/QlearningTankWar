from src.game.game import Game
from src.test import TestingEnvironment
from src.trainer import Trainer

if __name__ == "__main__":
    flag = "play"
    model_path = "models/version2_300.pth"
    if flag == 'train':
        trainer = Trainer(window_size=800)
        trainer.train(episodes=300)
    elif flag == 'play':
        game = Game()
        game.load_ai_model(model_path)
        game.run()
    elif flag == 'test':
        test = TestingEnvironment()
        test.load_latest_model(model_path)
        test.run()