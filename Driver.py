from tensorflow import keras
import tensorflow as tf
import numpy as np


def getRandomNumber():
    return 4


class Driver:

    def __init__(self):
        self.model_CNN = None
        pass

    # После инициализации модели. Необходимо подать картинку,
    # чтобы получить  вектор управления
    def Control_CNN(self, image: np.ndarray):
        # assert self.model_CNN is None, "model is None"
        return self.model_CNN.predict_proba([image[np.newaxis]]).argmax()  - 1  # Пока так!!!!!

    # Иннициализация модели
    def model_CNN_init(self, path):
        self.model_CNN = keras.models.load_model(path)
        self.model_CNN.summary()
        print(self.model_CNN)

    # Преобразование выходов сети
    def _convert_CNN(self, output):
        pass
