# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:37:22 2018

@author: EDSIM
Code for 2048 game from Tay Yang Shun
http://github.com/yangshun
"""
import numpy as np
import time
import tensorflow
import pandas as pd

from matplotlib import pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, InputLayer
from logic import *
from random import sample
from tqdm import tqdm



class Game:

    def __init__(self):
        self.matrix = new_game(4)

        self.matrix = add_two(self.matrix)
        self.matrix = add_two(self.matrix)
        self.score = 0
        self.commands = {0: up,
                         1: down,
                         2: left,
                         3: right}
        self.all_moves = []

    def get_flat_matrix(self, matrix):
        return np.array([[element for row in matrix for element in row]])

    def move(self, key):

        self.matrix, done, score = self.commands[key](self.matrix)
        self.score += score
        if done:
            self.matrix = add_two(self.matrix)
        return score

    def get_q_values(self, initial_matrix, model=None):
        if model is None:
            new_score = []
            for i in range(4):
                new_matrix, done, score = self.commands[i](initial_matrix)
                if new_matrix == initial_matrix:
                    score=-1
                new_score.append(score)
        else:
            new_score = model.predict(self.get_flat_matrix(initial_matrix))[0]
        return new_score

    def decide_and_move(self, epsilon=0, model=None):
        initial_matrix = self.get_flat_matrix(self.matrix)
        if np.random.random()<epsilon:
            move_to_make = np.random.choice(range(4))
        else:
            new_score = self.get_q_values(self.matrix, model=model)
            move_to_make = np.argmax(new_score)
        score = self.move(move_to_make)
        game_over = (game_state(self.matrix)=='lose')
        final_matrix = self.get_flat_matrix(self.matrix)
        return initial_matrix, move_to_make, score, final_matrix, game_over

    def play(self, epsilon=0, model=None):
        not_done = True
        while not_done:
            s0, a, r, s1, game_over = self.decide_and_move(epsilon=epsilon, model=None)
            self.all_moves.append((s0, a, r, s1, game_over))
            not_done = (game_over==0)

class Agent:

    def __init__(self, model_file=None):
        self.define_model(model_file)
        return

    def define_model(self, model_file, activation='linear'):
        model = Sequential()
        model.add(InputLayer(batch_input_shape=(1, 16)))
        model.add(Dense(64, activation='sigmoid'))
        model.add(Dense(4, activation=activation))
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        if model_file is not None:
            model.load_weights(model_file)
        self.model=model
        return model

    def train_incremental(increments, epochs=5, epsilon=.1, min_samples=10000, threshold=.8,
                          batch_size=512, decay=.9):
        self.memory = []
        self.scores = []        
        
        print('Generating initial data')
        for _ in tqdm(range(min_samples)):
            game = Game()
            game.play(epsilon=10, model=self.model)
            self.memory += game.all_moves
            self.scores.append(game.score)

        for i in range(increments):
            min_score = pd.Series(self.scores).quantile(threshold)
            self.memory = np.array(self.memory)[np.array(self.scores)>min_score].tolist()
            self.memory = np.random.shuffle(self.memory)

            X = np.array([m[0] for m in self.memory])
            y = np.array([m[1] for m in self.memory])
            self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

            print(f'{i}th training increment, minimum score {min_score}')

            self.scores = []            
            while len(self.scores)<min_samples:
                game = Game()
                game.play(model=self.model, epsilon=epsilon)
                if game.score>min_score:
                    self.memory += game.all_moves
                    self.scores.append(game.score)


    def train_q_learning(self, episodes, observe_episodes, epsilon, learning_rate,
                         decay, batch_size, threshold=2000):
        self.memory = []
        self.scores = []
        for episode in tqdm(range(episodes)):
            game = Game()
            game.play(epsilon=epsilon, model=self.model)
            if episode>=observe_episodes:
                minibatch = sample(self.memory, batch_size)
                for state, action, reward, next_state, done in minibatch:
                    target = reward
                    if not done:
                      target = (
                              reward + learning_rate *
                              np.amax(
                                      self.model.predict(next_state)[0]
                                      )
                              )
                    target_f = self.model.predict(state)
                    target_f[0][action] = target
                    self.model.fit(state, target_f, epochs=1, verbose=0)
                epsilon *= decay
            
            if ((episode//200-episode/200)==0):
                print(np.array(self.scores).mean())
                self.model.save('keras_2048_model')
            
            if game.score>threshold:
                self.scores.append(game.score)
                self.memory += game.all_moves


    def play_n_episodes_greedy(self, episodes, epsilon):
        self.scores = []
        max_numbers = []
        for _ in tqdm(range(episodes)):
            game = Game()
            game.play(epsilon=epsilon)
            self.scores.append(game.score)
            max_numbers.append(max(max(game.matrix)))
        print('Final max score', max(self.scores))
        print('Final max number', max(max_numbers))


if __name__=='__main__':

    agent = Agent() #model_file='keras_2018_model_25112018')
    # agent.play_n_episodes_greedy(10000, epsilon=10)
    agent.train_q_learning(episodes=50000,
                           observe_episodes=10000,
                           epsilon=.2,
                           learning_rate=.95,
                           decay=.9999,
                           batch_size=512)

    plt.hist(pd.Series(agent.scores), bins=50)
    plt.show()

    plt.plot(pd.Series(agent.scores))
    plt.plot(pd.Series(agent.scores).rolling(1000).mean())
    plt.show()

