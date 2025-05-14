import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import datetime

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from typing import Optional, Sequence
import sys
from skrl import logger


# seed for reproducibility
set_seed(41)  # e.g. `set_seed(42)` for fixed seed
now = datetime.datetime.now()
now_string = now.strftime('%Y_%m_%d_%H_%M')

def _print_cfg(d, indent=0) -> None:
    """Print the environment configuration

    :param d: The dictionary to print
    :type d: dict
    :param indent: The indentation level (default: ``0``)
    :type indent: int, optional
    """
    for key, value in d.items():
        if isinstance(value, dict):
            _print_cfg(value, indent + 1)
        else:
            print("  |   " * indent + f"  |-- {key}: {value}")


def load_isaaclab_env(
    task_name: str = "",
    num_envs: Optional[int] = None,
    headless: Optional[bool] = None,
    cli_args: Sequence[str] = [],
    show_cfg: bool = True,
):
    """Load an Isaac Lab environment

    Isaac Lab: https://isaac-sim.github.io/IsaacLab

    This function includes the definition and parsing of command line arguments used by Isaac Lab:

    - ``--headless``: Force display off at all times
    - ``--cpu``: Use CPU pipeline
    - ``--num_envs``: Number of environments to simulate
    - ``--task``: Name of the task
    - ``--num_envs``: Seed used for the environment

    :param task_name: The name of the task (default: ``""``).
                      If not specified, the task name is taken from the command line argument (``--task TASK_NAME``).
                      Command line argument has priority over function parameter if both are specified
    :type task_name: str, optional
    :param num_envs: Number of parallel environments to create (default: ``None``).
                     If not specified, the default number of environments defined in the task configuration is used.
                     Command line argument has priority over function parameter if both are specified
    :type num_envs: int, optional
    :param headless: Whether to use headless mode (no rendering) (default: ``None``).
                     If not specified, the default task configuration is used.
                     Command line argument has priority over function parameter if both are specified
    :type headless: bool, optional
    :param cli_args: Isaac Lab configuration and command line arguments (default: ``[]``)
    :type cli_args: list of str, optional
    :param show_cfg: Whether to print the configuration (default: ``True``)
    :type show_cfg: bool, optional

    :raises ValueError: The task name has not been defined, neither by the function parameter nor by the command line arguments

    :return: Isaac Lab environment
    :rtype: gymnasium.Env
    """
    import argparse
    import atexit
    import gymnasium

    # check task from command line arguments
    defined = False
    for arg in sys.argv:
        if arg.startswith("--task"):
            defined = True
            break
    # get task name from command line arguments
    if defined:
        arg_index = sys.argv.index("--task") + 1
        if arg_index >= len(sys.argv):
            raise ValueError(
                "No task name defined. Set the task_name parameter or use --task <task_name> as command line argument"
            )
        if task_name and task_name != sys.argv[arg_index]:
            logger.warning(f"Overriding task ({task_name}) with command line argument ({sys.argv[arg_index]})")
    # get task name from function arguments
    else:
        if task_name:
            sys.argv.append("--task")
            sys.argv.append(task_name)
        else:
            raise ValueError(
                "No task name defined. Set the task_name parameter or use --task <task_name> as command line argument"
            )

    # check num_envs from command line arguments
    defined = False
    for arg in sys.argv:
        if arg.startswith("--num_envs"):
            defined = True
            break
    # get num_envs from command line arguments
    if defined:
        if num_envs is not None:
            logger.warning("Overriding num_envs with command line argument (--num_envs)")
    # get num_envs from function arguments
    elif num_envs is not None and num_envs > 0:
        sys.argv.append("--num_envs")
        sys.argv.append(str(num_envs))

    # check headless from command line arguments
    defined = False
    for arg in sys.argv:
        if arg.startswith("--headless"):
            defined = True
            break
    # get headless from command line arguments
    if defined:
        if headless is not None:
            logger.warning("Overriding headless with command line argument (--headless)")
    # get headless from function arguments
    elif headless is not None:
        sys.argv.append("--headless")

    # others command line arguments
    sys.argv += cli_args

    # parse arguments
    parser = argparse.ArgumentParser("Isaac Lab: Omniverse Robotics Environments!")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
    parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes.")

    # launch the simulation app

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)



    @atexit.register
    def close_the_simulator():
        app_launcher.app.close()

    import isaaclab_tasks  # type: ignore
    from isaaclab_tasks.utils import parse_env_cfg  # type: ignore

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric)

    # multi-gpu training config
    if args.distributed:
        cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # print config
    if show_cfg:
        print(f"\nIsaac Lab environment ({args.task})")
        try:
            _print_cfg(cfg)
        except AttributeError as e:
            pass

    # load environment
    env = gymnasium.make(args.task, cfg=cfg, render_mode="rgb_array" if args.video else None)



    return env



class resblock(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1, size=[112, 112]):
        super(resblock, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, 3, stride=stride, padding=1)
        self.ln1 = nn.LayerNorm([math.ceil(size[0] / stride), math.ceil(size[1] / stride)])
        self.conv2 = nn.Conv2d(ch_out, ch_out, 3, stride=1, padding=1)
        self.ln2 = nn.LayerNorm([math.ceil(size[0] / stride), math.ceil(size[1] / stride)])

        self.extra = nn.Sequential()
        if ch_in != ch_out and stride == 1:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, 1, 1),
                nn.LayerNorm([size[0], size[1]])
            )
        elif ch_in != ch_out and stride == 2:
            self.extra = nn.Sequential(
                nn.MaxPool2d(2, stride=stride),
                nn.Conv2d(ch_in, ch_out, 1, 1),
                nn.LayerNorm([math.ceil(size[0] / stride), math.ceil(size[1] / stride)])
            )

    def forward(self, x):
        out = F.relu(self.ln1(self.conv1(x)))
        out = self.ln2(self.conv2(out))

        x = self.extra(x)

        out = out + x
        out = F.relu(out)

        return out

# define shared model (stochastic and deterministic models) using mixins
class CNN_resnet10(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        # input 224*224
        # self.features_extractor = nn.Sequential(nn.Conv2d(1, 8, 5, 2, 2),
        #                          nn.LayerNorm([112, 112]),
        #                          nn.ReLU(),
        #                          resblock(8, 16, stride=2, size=[112, 112]),
        #                          resblock(16, 32, stride=2, size=[56, 56]),
        #                          resblock(32, 64, stride=1, size=[28, 28]),
        #                          resblock(64, 64, stride=1, size=[28, 28]),
        #                          nn.Conv2d(64, 1, 1, 1),
        #                          nn.Flatten(start_dim=1))

        # input 112*112
        self.features_extractor = nn.Sequential(nn.Conv2d(1, 8, 3, 1, 1),
                                                nn.LayerNorm([112, 112]),
                                                nn.ReLU(),
                                                resblock(8, 16, stride=2, size=[112, 112]),
                                                resblock(16, 32, stride=2, size=[56, 56]),
                                                resblock(32, 64, stride=1, size=[28, 28]),
                                                resblock(64, 64, stride=1, size=[28, 28]),
                                                nn.Conv2d(64, 1, 1, 1),
                                                nn.Flatten(start_dim=1))

        self.state_extractor = nn.Sequential(nn.Flatten(start_dim=1))

        self.net = nn.Sequential(nn.Linear(784+12, 128),
                                 # nn.BatchNorm1d(128),
                                 nn.Tanh(),
                                 nn.Linear(128, 64),
                                 # nn.BatchNorm1d(64),
                                 nn.Tanh(),
                                 nn.Linear(64, 32),
                                 # nn.BatchNorm1d(32),
                                 nn.Tanh())

        self.mean_layer = nn.Linear(32, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Linear(32, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        input = inputs["states"].reshape([-1, *self.observation_space.shape])
        cnn_feature = self.features_extractor(input.permute(0, 3, 1, 2)[:,:,:-1,:])
        state_feature = self.state_extractor(input.permute(0, 3, 1, 2)[:,:,-1,:])
        mix_feature = torch.cat([cnn_feature, state_feature[:, :12]], dim=1)
        if role == "policy":
            self._shared_output = self.net(mix_feature)
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role == "value":
            shared_output = self.net(mix_feature) if self._shared_output is None else self._shared_output
            self._shared_output = None
            return self.value_layer(shared_output), {}


# load and wrap the Isaac Lab environment
task_name = "Isaac-Quadcopterend2end-Direct-v0"
env = load_isaaclab_env(task_name=task_name, num_envs=10, cli_args=["--distributed", "--enable_cameras"])
env = wrap_env(env)

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=24, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
# models["policy"] = Shared(env.observation_space, env.action_space, device)
models["policy"] = CNN_resnet10(env.observation_space, env.action_space, device)
models["value"] = models["policy"]  # same instance: shared model


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 24  # memory_size
cfg["learning_epochs"] = 5
cfg["mini_batches"] = 4  # 24 * 4096 / 24576
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-3
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 1.0
cfg["kl_threshold"] = 0
cfg["rewards_shaper"] = None
cfg["time_limit_bootstrap"] = False
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 600
cfg["experiment"]["checkpoint_interval"] = 6000
cfg["experiment"]["directory"] = "runs/torch/" + task_name
cfg["experiment"]["wandb"] = False
cfg["experiment"]["wandb_kwargs"] = {
    "entity": "siwufei",
    "project": "IsaacLab",
    "name": now_string + "_" + task_name,
}

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

# Load the checkpoint
agent.load("/home/li/Desktop/Quadcopter_Lab/runs/torch/Isaac-Quadcopterend2end-Direct-v0/25-04-08_10-41-23-232636_PPO/checkpoints/best_agent.pt")

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 4800000, "headless": False}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()
