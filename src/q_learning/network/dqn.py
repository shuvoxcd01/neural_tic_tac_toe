import os

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Flatten
from tensorflow.keras.models import clone_model


class DQN:
    @staticmethod
    def get_q_network(input_shape, num_actions):
        model = Sequential()
        model.add(InputLayer(input_shape=input_shape))
        model.add(Flatten())
        model.add(Dense(units=100, activation="relu"))
        model.add(Dense(units=250, activation="relu"))
        model.add(Dense(units=100, activation='relu'))
        model.add(Dense(units=50, activation='relu'))
        model.add(Dense(units=num_actions))

        return model

    @staticmethod
    def clone(model):
        cloned_model = clone_model(model=model)
        cloned_model.set_weights(model.get_weights())

        return cloned_model

    @staticmethod
    def save_model(model, saved_model_dir, saved_model_name):
        if not os.path.exists(saved_model_dir):
            os.makedirs(saved_model_dir)

        path_to_saved_model = os.path.join(saved_model_dir, saved_model_name)

        model.save(path_to_saved_model)
