import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.graph_objects as go

class TradingEnvPlotter:
    def __init__(self, env):
        self.env = env

    def plot_history(self):
        df_history = pd.DataFrame(self.env.history)
        df_history.set_index('step', inplace=True)

        fig, ax = plt.subplots(2, 2, figsize=[8, 6])

        # Total Asset over time
        ax[0][0].plot(self.env.total_asset_history, label='Total Asset', linestyle='-')
        ax[0][0].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
        ax[0][0].set_title('Total Asset over Time')
        ax[0][0].legend(loc='upper left')

        # Action over time
        ax[0][1].plot(self.env.actions_history, label='Actions History', linestyle='-')
        ax[0][1].set_title('Action over time')
        ax[0][1].legend(loc='upper left')
        ax[0][1].set_yticks(range(self.env.action_space.n))

        # Cumulative reward plot
        ax[1][0].plot(np.cumsum(self.env.episode_rewards), label='Cumulative Reward', linestyle='-')
        ax[1][0].set_title('Cumulative Reward per Episode')
        ax[1][0].legend(loc='upper left')
        ax[1][0].autoscale_view(scalex=True, scaley=True)

        # Reward over time
        ax[1][1].plot(self.env.episode_rewards, label='Reward per step', linestyle='-')
        ax[1][1].set_title('Reward per Step')
        ax[1][1].legend(loc='upper left')
        ax[1][1].autoscale_view(scalex=True, scaley=True)

        plt.tight_layout()
        plt.show()

    def plot_stock_price(self):
        fig, ax = plt.subplots(figsize=[16, 4])
        ax.plot(self.env.df.index, self.env.df['close'], label='Close Price')
        ax.set_title('Stock Price')
        ax.legend(loc='upper left')
        plt.tight_layout()
        plt.show()