"""
Launcher for experiments with PEARL

"""
import os
import pathlib
import numpy as np
import click
import json
import torch

from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.sac import PEARLSoftActorCritic
from rlkit.torch.sac.agent import PEARLAgent
from rlkit.launchers.launcher_util import run_experiment
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config
import rlkit.core.hyperparameter as hyp

import click
import json
import os

import rlkit.pythonplusplus as ppp


def pearl_experiment(variant):
    from rlkit.envs import ENVS, register_pearl_envs
    register_pearl_envs()
    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))


    # create multi-task environment and sample tasks
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params']['use_next_obs_in_context'] else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    context_encoder = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
    )
    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    reward_predictor = FlattenMlp(
        hidden_sizes=[200, 200, 200],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        reward_predictor,
        **variant['algo_params']
    )
    algorithm = PEARLSoftActorCritic(
        env=env,
        train_tasks=list(tasks[:variant['n_train_tasks']]),
        eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
        nets=[agent, qf1, qf2, vf],
        latent_dim=latent_dim,
        **variant['algo_params']
    )

    # optionally load pre-trained weights
    if variant['path_to_weights'] is not None:
        path = variant['path_to_weights']
        context_encoder.load_state_dict(torch.load(os.path.join(path, 'context_encoder.pth')))
        qf1.load_state_dict(torch.load(os.path.join(path, 'qf1.pth')))
        qf2.load_state_dict(torch.load(os.path.join(path, 'qf2.pth')))
        vf.load_state_dict(torch.load(os.path.join(path, 'vf.pth')))
        # TODO hacky, revisit after model refactor
        algorithm.networks[-2].load_state_dict(torch.load(os.path.join(path, 'target_vf.pth')))
        policy.load_state_dict(torch.load(os.path.join(path, 'policy.pth')))

    # optional GPU mode
    # ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    # if ptu.gpu_enabled():
    #     algorithm.to()
    algorithm.to(ptu.device)

    # create logging directory
    # TODO support Docker
    # if exp_name == '':
    #     exp_name = variant['env_name']
    # if DEBUG:
    #     exp_id = 'debug'
    #     exp_name = 'dev-' + exp_name
    # else:
    #     exp_id = None
    # experiment_log_dir = setup_logger(
    #     exp_name,
    #     variant=variant, exp_id=exp_id, base_log_dir=variant['util_params']['base_log_dir'])

    # optionally save eval trajectories as pkl files
    # if variant['algo_params']['dump_eval_paths']:
    #     pickle_dir = experiment_log_dir + '/eval_trajectories'
    #     pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # run the algorithm
    algorithm.train()






@click.command()
@click.argument('config', default=None)
@click.option('--debug', is_flag=True, default=False)
@click.option('--exp_name', default=None)
@click.option('--mode', default='local')
@click.option('--gpu', default=False)
@click.option('--nseeds', default=1)
def main(config, debug, exp_name, mode, gpu, nseeds):
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
    else:  # You can also hard-code it if you don't give a config file.
        exp_params = {}
    if debug:
        exp_params['algo_params'] = {
            "meta_batch": 4,
            "num_initial_steps": 20,
            "num_steps_prior": 10,
            "num_steps_posterior": 0,
            "num_extra_rl_steps_posterior": 10,
            "num_evals": 4,
            "num_steps_per_eval": 6,
            "num_exp_traj_eval": 2,
            "embedding_batch_size": 256,
            "num_iterations_with_reward_supervision": 10,
            "freeze_encoder_buffer_in_unsupervised_phase": True,
            "train_reward_pred_in_unsupervised_phase": False,
            "embedding_mini_batch_size": 256,
            "num_train_steps_per_itr": 20,
            "max_path_length": 2,
        }
        exp_params["net_size"] = 3
    variant = ppp.merge_recursive_dicts(
        exp_params,
        default_config,
        ignore_duplicate_keys_in_second_dict=True,
    )

    mode = mode or 'local'
    exp_name = exp_name or 'dev'

    search_space = {
        'algo_params.num_iterations_with_reward_supervision': [
            # 10,
            # 20,
            30,
            100,
            200,
            9999,
        ],
        'algo_params.freeze_encoder_buffer_in_unsupervised_phase': [
            True,
            # False,
        ],
        'algo_params.train_reward_pred_in_unsupervised_phase': [
            # True,
            False,
        ],
        'algo_params.use_encoder_snapshot_for_reward_pred_in_unsupervised_phase': [
            True,
            # False,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(nseeds):
            variant['exp_id'] = exp_id
            run_experiment(
                pearl_experiment,
                unpack_variant=False,
                exp_name=exp_name,
                mode=mode,
                variant=variant,
                time_in_mins=int(2.8 * 24 * 60),  # if you use mode=sss
                use_gpu=gpu,
            )
    print(exp_name)


if __name__ == "__main__":
    main()

