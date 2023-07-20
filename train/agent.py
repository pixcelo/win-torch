import numpy as np
import random
import torch
import torch.nn as nn
from torch import optim
from collections import deque
from qnetwork import QNetwork
import datetime

class Agent:
    def __init__(self, env, gamma=0.99, lr=0.01, batch_size=64, memory_size=10000):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size

        self.action_size = self.env.action_space.n
        self.obs_size = self.env.observation_space.shape[0]
        self.memory = deque(maxlen=memory_size)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.model = QNetwork(self.obs_size, self.action_size)
        self.target_model = QNetwork(self.obs_size, self.action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = torch.FloatTensor(state)
        with torch.no_grad():
            act_values = self.model(state)
        return np.argmax(act_values.cpu().numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = self.model(torch.FloatTensor(state))
            if done:
                target[action] = reward
            else:
                target_val = self.target_model(torch.FloatTensor(next_state))
                target[action] = reward + self.gamma * torch.max(target_val).item()

            self.optimizer.zero_grad()
            output = self.model(torch.FloatTensor(state))
            loss = self.criterion(output, target.detach())
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def learn(self, current_episode, total_episodes):
        state = self.env.reset()
        state = np.reshape(state, [1, self.obs_size])
        for time in range(total_episodes):
            action = self.choose_action(state[0])
            next_state, reward, done, _ = self.env.step(action)
            next_state = np.reshape(next_state, [1, self.obs_size])
            self.remember(state[0], action, reward, next_state[0], done)
            state = next_state
            if done:                
                break
            if len(self.memory) > self.batch_size:
                self.replay()
            
        return self.env.episode_reward

    def save_model(self):
        now = datetime.datetime.now()
        now_str = now.strftime('%Y%m%d_%H%M%S')
        torch.save(self.q_network.state_dict(), f'../model/model_weights_{now_str}.pth')

    def load_model(self, model_path):
        self.q_network.load_state_dict(torch.load(model_path))
        self.q_network.eval()