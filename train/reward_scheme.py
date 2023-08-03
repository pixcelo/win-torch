class RewardScheme:
    def __init__(self):
        # 連勝、連敗のカウンタ
        self.win_streak = 0
        self.loss_streak = 0

    def reset(self):
        self.win_streak = 0
        self.loss_streak = 0

    def get_reward(self, prev_balance, current_balance, action_taken):
        # 基本報酬の計算
        immediate_reward = current_balance - prev_balance
        
        # 勝利ボーナス・連勝ボーナスの計算
        if immediate_reward > 0:
            self.win_streak += 1
            self.loss_streak = 0
            win_bonus = 0.1  # 一回の勝利に対するボーナス
            streak_bonus = 0.05 * self.win_streak  # 連勝ボーナス
        else:
            self.loss_streak += 1
            self.win_streak = 0
            win_bonus = 0
            streak_bonus = 0

        # 損失ペナルティの計算 (オプション)
        # loss_penalty = -0.05 * self.loss_streak if immediate_reward < 0 else 0
        
        # 全体の報酬
        total_reward = immediate_reward + win_bonus + streak_bonus # + loss_penalty

        return total_reward