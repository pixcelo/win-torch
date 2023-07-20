from gym import spaces
from trading_env import TradingEnv

class ShortingTradingEnv(TradingEnv):
    def __init__(self, df, initial_cash=100000, transaction_cost=0.001, max_holdings=1):
        super(ShortingTradingEnv, self).__init__(df, initial_cash, transaction_cost, max_holdings)

        self.action_space = spaces.Discrete(5)  # 0: Do nothing, 1: Buy, 2: Sell, 3: Short, 4: Cover

    def step(self, action):
        assert self.action_space.contains(action)

        current_price = self.df.iloc[self.current_step]['close']
        done = self.current_step == len(self.df) - 1
        if done:
            return self._get_observation(), 0, done, {}

        self.current_step += 1
        old_total_asset = self.cash + self.holdings * current_price
        reward = 0
        penalty = 0.01

        if action == 1:  # Buy
            if self.cash >= current_price and self.holdings < self.max_holdings:
                self.holdings += 1
                self.cash -= current_price * (1 + self.transaction_cost)
            else:
                reward -= old_total_asset * penalty
        elif action == 2:  # Sell
            if self.holdings > 0:
                self.holdings -= 1
                self.cash += current_price * (1 - self.transaction_cost)
            else:
                reward -= old_total_asset * penalty
        elif action == 3:  # Short
            if self.cash >= current_price and self.holdings > -self.max_holdings:  # Adjust here to allow shorting
                self.holdings -= 1  # Shorting increase the negative holdings
                self.cash += current_price * (1 - self.transaction_cost)  # And increase the cash
            else:
                reward -= old_total_asset * penalty
        elif action == 4:  # Cover
            if self.holdings < 0:  # Only can cover if there are shorts
                self.holdings += 1  # Covering decrease the negative holdings
                self.cash -= current_price * (1 + self.transaction_cost)  # And decrease the cash
            else:
                reward -= old_total_asset * penalty

        # If no stocks are held and no action is taken, reward is zero due to lost opportunity.
        if self.holdings == 0 and action == 0:
            reward -= old_total_asset * penalty

        # Reward is the relative change in total assets, with penalties for invalid actions and inaction when no stocks are held.
        new_total_asset = self.cash + self.holdings * current_price
        reward += (new_total_asset - old_total_asset) / old_total_asset if old_total_asset > 0 else 0
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

