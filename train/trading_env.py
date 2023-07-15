import gym
import numpy as np
import pandas as pd
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, df, initial_cash=1000000, transaction_cost=0.001, max_holdings=1):
        super(TradingEnv, self).__init__()

        self.df = df
        self.reward_range = (-np.inf, np.inf)
        self.action_space = spaces.Discrete(3)  # 0: Do nothing, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.df.shape[1] + 4,))

        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.holdings = 0
        self.max_holdings = max_holdings
        self.episode_reward = 0
        self.episode_rewards = []
        self.actions_history = []
        self.total_asset_history = []
        self.reset()

    def step(self, action):
        assert self.action_space.contains(action)

        current_price = self.df.iloc[self.current_step]['close']
        done = self.current_step == len(self.df) - 1
        if done:
            return self._get_observation(), 0, done, {}

        self.current_step += 1
        old_total_asset = self.cash + self.holdings * current_price
        reward = 0
        penalty = 0.1

        if action == 1:  # Buy
            if self.cash >= current_price and self.holdings < self.max_holdings:
                self.holdings += 1
                self.cash -= current_price * (1 + self.transaction_cost)
            else:
                reward -= penalty
        elif action == 2:  # Sell
            if self.holdings > 0:
                self.holdings -= 1
                self.cash += current_price * (1 - self.transaction_cost)
            else:
                reward -= penalty

        # If no stocks are held and no action is taken, reward is zero due to lost opportunity.
        if self.holdings == 0 and action == 0:
            reward = -0.1

        new_total_asset = self.cash + self.holdings * current_price
        reward += new_total_asset - old_total_asset
        self.episode_reward += reward

        # Record the history
        self.history.append({
            "step": self.current_step,
            "cash": self.cash,
            "action": action,
            "total_asset": new_total_asset,
            "reward": reward,
            "done": done
        })
        self.episode_rewards.append(self.episode_reward)
        self.actions_history.append(action)
        self.total_asset_history.append(new_total_asset)

        return self._get_observation(), reward, done, {}

    def reset(self):
        self.cash = self.initial_cash
        self.holdings = 0
        self.current_step = 0
        self.history = []
        return self._get_observation()
    
    def _get_observation(self):
        total_asset = self.cash + self.holdings * self.df.iloc[self.current_step]['close']
        cash_ratio = self.cash / total_asset if total_asset > 0 else 0
        holdings_ratio = self.holdings * self.df.iloc[self.current_step]['close'] / total_asset if total_asset > 0 else 0
        return np.append(self.df.iloc[self.current_step], [self.holdings, self.cash, cash_ratio, holdings_ratio])
