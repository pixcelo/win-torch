import gym
import numpy as np
import pandas as pd
from gym import spaces
from reward_calculator import RewardCalculator
from reward_scheme import RewardScheme

class TradingEnv(gym.Env):
    def __init__(self, df, initial_cash=100000, transaction_cost=0.001, max_holdings=1):
        super(TradingEnv, self).__init__()

        self.df = df
        self.reward_range = (-np.inf, np.inf)
        self.action_space = spaces.Discrete(3)  # 0: Do nothing, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.df.shape[1] + 5,))

        self.reward_calculator = RewardCalculator(max_length=30)
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.holdings = 0
        self.max_holdings = max_holdings
        self.cumulative_rewards = 0
        self.cumulative_rewards_squared = 0
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

        if action == 1:  # Buy
            if self.cash >= current_price and self.holdings < self.max_holdings:
                self.holdings += 1
                self.cash -= current_price * (1 + self.transaction_cost)

        elif action == 2:  # Sell
            if self.holdings > 0:
                self.holdings -= 1
                self.cash += current_price * (1 - self.transaction_cost)

        new_total_asset = self.cash + self.holdings * current_price
        
        # Use RewardCalculator to calculate reward
        # reward = self.reward_calculator.calculate(new_total_asset)
        reward_calculator = RewardScheme()
        reward = reward_calculator.get_reward(old_total_asset, new_total_asset, action)
        
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
        
        N = len(self.history) + 1  # the current number of steps
        if N == 1:
            sharpe_ratio = 0.0
        else:
            mean_rewards = self.cumulative_rewards / N
            mean_rewards_squared = self.cumulative_rewards_squared / N
            variance_rewards = mean_rewards_squared - mean_rewards**2
            std_rewards = np.sqrt(variance_rewards)
            sharpe_ratio = mean_rewards / std_rewards if std_rewards > 0 else 0.0

        return np.append(self.df.iloc[self.current_step], [self.holdings, total_asset, cash_ratio, holdings_ratio, sharpe_ratio])
