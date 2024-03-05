import argparse


class Args():
    def __init__(self):
        parser = argparse.ArgumentParser(description="Pytorch RL algorithms")
        parser.add_argument(
            "--seed", type=int, default=10, help="random seed for reproducibility"
        )
        parser.add_argument(
            "--envname", type=str, default="HalfCheetah-v4"   #  "BipedalWalker-v3" "Pendulum-v1"   Mujoco: 'HalfCheetah-v4', Humanoid-v4, Swimmer-v4, Ant-v2, InvertedPendulum-v4, Walker2d-v4, LunarLander-v2
        )
        parser.add_argument(
            "--test", type=bool, default=False
        )
        parser.add_argument(
            "--getfig", type=bool, default=False
        )
        parser.add_argument(
            "--suffix", type=str, default='9',
        )
        parser.add_argument("--algo", type=str, default="MaTD3", help="choose an algorithm") # ="MaTD3"
        # Conti Action
        parser.add_argument(
            "--off-render", dest="render", action="store_false", help="turn off rendering"
        )
        parser.add_argument(
            "--interim-test-num", type=int, default=1, help="interim test number"
        )
        """ Replay Buffer Setup """

        parser.add_argument(
            "--fixbeta", action="store_true",
        )
        parser.add_argument(
            "--index", type=float, default=0.4,
        )
        parser.add_argument(
            "--savefolder", type=str, default='test', help="save_folder_for_csv"
        )
        parser.add_argument(
            "--evalstep", type=int, default=5000,
        )
        parser.add_argument(
            "--savealgname", type=str, default='',
        )
        args = parser.parse_args()

        self.args = args
        if self.args.envname == 'Pendulum-v1':
            self.args.max_episode_steps = 200
        elif self.args.envname == 'LunarLanderContinuous-v2':
            self.args.max_episode_steps = 1000
        elif self.args.envname == 'BipedalWalker-v3':
            self.args.max_episode_steps = 1600
        elif self.args.envname == 'BipedalWalkerHardcore-v3':
            self.args.max_episode_steps = 2000
        else:
            self.args.max_episode_steps = 1000

    def return_arg(self):
        return self.args


args = Args()
args = args.return_arg()
