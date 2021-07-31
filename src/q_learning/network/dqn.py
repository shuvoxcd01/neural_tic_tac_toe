from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import clone_model


class DQN:
    @staticmethod
    def get_q_network(input_shape, num_actions):
        model = Sequential()
        model.add(Dense(units=100, input_shape=input_shape, activation='relu'))
        model.add(Dense(units=50, activation='relu'))
        model.add(Dense(units=num_actions))

        return model

    @staticmethod
    def clone(model):
        cloned_model = clone_model(model=model)
        cloned_model.set_weights(model.get_weights())

        return cloned_model
