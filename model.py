from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


def build_model(num_states, num_actions, lr):
    model = Sequential()
    model.add(Dense(64, input_dim=num_states, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=lr))
    return model
