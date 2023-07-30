class RewardCalculator:
    def __init__(self, max_length=100):
        # max_length is the length of the period to calculate normalized asset fluctuations.
        self.asset_values = []
        self.max_length = max_length
        self.max_drawdown = 0
        self.highest_value = 0

    def calculate(self, current_value):
        if len(self.asset_values) > 0:
            prev_value = self.asset_values[-1]
        else:
            prev_value = current_value  # Use the current value as the previous value for the first time

        normalized_return = (current_value - prev_value) / prev_value
        drawdown = (self.highest_value - current_value) / self.highest_value if self.highest_value > 0 else 0

        # Update the maximum drawdown
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        # Update the highest value
        if current_value > self.highest_value:
            self.highest_value = current_value

        # Add the latest asset value and remove old data
        self.asset_values.append(current_value)
        if len(self.asset_values) > self.max_length:
            self.asset_values.pop(0)

        # Asset fluctuation normalized over a period
        if len(self.asset_values) > 1:
            avg_return = (self.asset_values[-1] - self.asset_values[0]) / self.asset_values[0]
        else:
            avg_return = 0

        # Penalty for maximum drawdown
        drawdown_penalty = -self.max_drawdown

        # Final reward
        final_reward = avg_return + drawdown_penalty

        return final_reward

