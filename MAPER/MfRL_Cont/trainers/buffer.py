class Buffer():
    def __init__(self, args,
                 buffersize,
                 batchsize,
                 alpha,
                 state_dim,
                 action_dim,
                 gamma):
        self.alpha = alpha
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batchsize = batchsize
        self.buffersize = buffersize
        self.args = args
        self.gamma = gamma

        if 'MaSAC' in self.args.algo or 'MaTD3' in self.args.algo:
            from algorithms.common.buffer.maper import MaPER
            self.memory = MaPER(
                self.args.algo,
                self.buffersize,
                self.batchsize,
                alpha=self.alpha,
                gamma=self.gamma
            )
        elif 'PER' in self.args.algo:
            from algorithms.common.buffer.per import PER
            self.memory = PER(
                self.args.algo,
                self.buffersize,
                self.batchsize,
                alpha=self.alpha,
                gamma=self.gamma
            )
        elif 'RANDOM' or 'TD3' in self.args.algo:
            from algorithms.common.buffer.replay_buffer import ReplayBuffer
            self.memory = self.memory = ReplayBuffer(
                self.buffersize, self.batchsize,
                gamma=self.gamma
            )
        else:
            assert False

    def return_buffer(self):

        return self.memory
