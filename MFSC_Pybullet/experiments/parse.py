import argparse


def get_args():
    parser = argparse.ArgumentParser(description='parameters for training and visualization')
    parser.add_argument('--algo', '-a', type=str, help='algorithm',
                        choices=['ppopixel', 'pporad', 'ppokeypoint'])
    parser.add_argument('--task', '-t', type=str, help='task to perform', required=True)
    parser.add_argument('--obs', choices=['hybrid', 'state'], default='hybrid', help='observation type')
    parser.add_argument('--env_version', '-v', type=int, help='version of environment', required=True)
    parser.add_argument('--exp_id', '-e', type=str, help='experiment id', required=True)
    parser.add_argument('--exp_name', type=str, default='', help='experiment id')
    parser.add_argument('--augment', action='store_true', help='enable random crop augmentation')
    parser.add_argument('--frame_stack', type=int, default=1, help='how many frames to stack')
    parser.add_argument('--use_hybrid_feature', '-u', action='store_true', help='Include joint states to policy')
    parser.add_argument('--latent_stack', '-l', action='store_true', help='share encoder for each frame in frame stack')

    parser.add_argument('--num_envs', type=int, default=16, help='number of parallel environments for ppo')
    parser.add_argument('--total_timesteps', type=int, default=int(5e6), help='Number of total env steps to train')
    parser.add_argument('--save_freq', type=int, default=1000000, help='Save frequency in env steps')
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed for environments')

    # PPO Common
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--buffer_size', type=int, default=int(1e5), help='obs/replay buffer size')
    parser.add_argument('--n_steps', type=int, help='Number of steps to run for each environment per update')
    parser.add_argument('--batch_size', type=int, help='Minibatch size')
    parser.add_argument('--n_epochs', type=int, help='Number of steps for each environment per update ')
    parser.add_argument('--gamma', type=float, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, help='Factor for trade-off  for Generalized Advantage Estimator')
    parser.add_argument('--clip_range', type=float, help='ppo clip parameter')
    parser.add_argument('--clip_range_vf', type=float, help='ppo clip parameter')
    parser.add_argument('--ent_coef', type=float, help='entropy term coefficient')
    parser.add_argument('--vf_coef', type=float, help='Value function coefficient')
    parser.add_argument('--max_grad_norm', type=float, help='The maximum value for the gradient clipping ')
    parser.add_argument('--target_kl', type=float, help='The maximum value for the gradient clipping ')

    args = parser.parse_args()

    return args
