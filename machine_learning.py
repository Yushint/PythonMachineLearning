import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import load_model
from PIL import Image
import numpy as np
import os.path


(TRAIN_DATA, TRAIN_TARGET), (TEST_DATA, TEST_TARGET) = mnist.load_data()
image_rows, image_cols = 28, 28

# входные данные свёрточной нейросети - 4-мерный тензор.
TRAIN_DATA = TRAIN_DATA.reshape(TRAIN_DATA.shape[0], image_rows, image_cols, 1)
TEST_DATA = TEST_DATA.reshape(TEST_DATA.shape[0], image_rows, image_cols, 1)
INPUT_DATA_SHAPE = (image_rows, image_cols, 1)

TRAIN_DATA = TRAIN_DATA.astype("float32") / 255.0
TEST_DATA = TEST_DATA.astype("float32") / 255.0

TRAIN_TARGET = to_categorical(TRAIN_TARGET)
TEST_TARGET = to_categorical(TEST_TARGET)

OUTPUT_CLASSES_NUMBER = len(TRAIN_TARGET[0]) # 10 classes 

def main():
    """Главная функция программы."""
    model = build_model()
    fit_and_evaluate_model(model)

def build_model():
    """Функция создаёт свёрточную ML-модель."""
    if check_if_file_exists("./machine_learning_model.h5"):
        model = load_model("machine_learning_model.h5")
    else:
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(3,3),
                         activation="relu",
                         input_shape=INPUT_DATA_SHAPE))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(32, (3,3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(OUTPUT_CLASSES_NUMBER, activation="softmax"))
    return model

def fit_and_evaluate_model(model):
    """ Обучение и валидация модели."""
    if check_if_file_exists("./machine_learning_model.h5"):
        print("Модель инициализирована.")
    else:    
        model.summary()
        model.compile(loss="categorical_crossentropy",
                  optimizer="rmsprop",
                  metrics=["accuracy"])
        model.fit(TRAIN_DATA, TRAIN_TARGET, epochs=2, batch_size=200)
        score = model.evaluate(TEST_DATA, TEST_TARGET, verbose=1)
        model.save("machine_learning_model.h5")
        print("Модель сохранена под именем machine_learning_model.h5 .")
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
        
def check_if_file_exists(full_filename:str) -> bool:
    """Проверяет, существует ли файл в директории."""
    return os.path.exists(full_filename)

def find_max_by_iteration(array) -> tuple:
    """Находит максимальный элемент и индекс максимального элемента в 
       итерируемой коллекции.
    """
    max_index = 0
    max_element = array[0]
    for i in range(1, len(array)):
        if array[i] > max_element:
            max_element = array[i]
            max_index = i
    return (max_element, max_index)
    
if __name__ == "__main__":
    main()
