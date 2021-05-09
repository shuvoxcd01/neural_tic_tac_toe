from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD


class DQN:
    def __call__(self, input_shape, num_actions):
        model = Sequential()
        model.add(Dense(units=24, input_shape=input_shape, activation='relu'))
        model.add(Dense(units=24, activation='relu'))
        model.add(Dense(units=num_actions, activation='linear'))

        model.compile(loss="mse", optimizer=SGD())

        return model
