import argparse


def run(env, args):
    """Run training or test.

    Args:
        env (gym.Env): openAI Gym environment with continuous action space
        args (argparse.Namespace): arguments including training settings
        state_dim (int): dimension of states
        action_dim (int): dimension of actions

    """
    # create an agent
    if 'MaSAC' in args.algo:
        from algorithms.sac.masac import MaSAC
        agent = MaSAC(env, args)
    elif 'MaTD3' in args.algo:
        from algorithms.td3.matd3 import MaTD3
        agent = MaTD3(env, args)
    elif 'TD3' in args.algo:
        from algorithms.td3.td3 import TD3Agent
        agent = TD3Agent(env, args)
    elif 'SAC' in args.algo:
        from algorithms.sac.sac import SACAgent
        agent = SACAgent(env, args)

    print(args.algo)
    agent.train()
