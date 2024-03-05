class EnvSampler():
    def __init__(self, env, max_path_length=1000):
        self.env = env
        self.path_length = 0
        self.current_state = None
        self.max_path_length = max_path_length
        self.path_rewards = []
        self.sum_reward = 0
        self.count = 0

    def sample(self, agent, eval_t=False):
        if self.current_state is None:
            self.current_state = self.env.reset()
            self.count = 0

        cur_state = self.current_state
        action = agent.select_action(self.current_state, eval_t)
        next_state, reward, terminal, info = self.env.step(action)
        self.path_length += 1
        self.sum_reward += reward
        self.count += 1

        # TODO: Save the path to the env_pool
        if terminal or self.path_length >= self.max_path_length:
            self.current_state = None
            self.path_length = 0
            self.path_rewards.append(self.sum_reward)
            self.sum_reward = 0
        else:
            self.current_state = next_state
        if self.count == 1000:
            terminal = True
            self.count = 0
            self.current_state = None
        return cur_state, action, next_state, reward, terminal, info
