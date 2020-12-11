from rlkit.envs.ant_dir import AntDirEnv
from rlkit.envs.ant_goal import AntGoalEnv
from rlkit.envs.half_cheetah_dir import HalfCheetahDirEnv
from rlkit.envs.half_cheetah_vel import HalfCheetahVelEnv
from rlkit.envs.humanoid_dir import HumanoidDirEnv
from rlkit.envs.point_robot import PointEnv, SparsePointEnv
# from rlkit.envs.hopper_rand_params_wrapper import \
#     HopperRandParamsWrappedEnv
# from rlkit.envs.walker_rand_params_wrapper import \
#     WalkerRandParamsWrappedEnv

ENVS = {}


def register_env(name):
    """Registers a env by name for instantiation in rlkit."""

    def register_env_fn(fn):
        if name in ENVS:
            raise ValueError("Cannot register duplicate env {}".format(name))
        if not callable(fn):
            raise TypeError("env {} must be callable".format(name))
        ENVS[name] = fn
        return fn

    return register_env_fn

def _register_env(name, fn):
    """Registers a env by name for instantiation in rlkit."""
    if name in ENVS:
        raise ValueError("Cannot register duplicate env {}".format(name))
    if not callable(fn):
        raise TypeError("env {} must be callable".format(name))
    ENVS[name] = fn


def register_pearl_envs():
    _register_env('sparse-point-robot', SparsePointEnv)
    _register_env('ant-dir', AntDirEnv)
    _register_env('ant-goal', AntGoalEnv)
    _register_env('cheetah-dir', HalfCheetahDirEnv)
    _register_env('cheetah-vel', HalfCheetahVelEnv)
    _register_env('humanoid-dir', HumanoidDirEnv)
    _register_env('point-robot', PointEnv)
    # _register_env('walker-rand-params', WalkerRandParamsWrappedEnv)
    # _register_env('hopper-rand-params', HopperRandParamsWrappedEnv)
