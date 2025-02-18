from src.game.game import Game
from src.trainer import Trainer

if __name__ == "__main__":
    flag = "play"

    if flag == 'train':
        trainer = Trainer(window_size=800)
        trainer.train(episodes=300)
    elif flag == 'play':
        model_path = "models/version2_300.pth"

        game = Game()
        game.load_ai_model(model_path)
        game.run()