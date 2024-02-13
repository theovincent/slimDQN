import json

DEFAULT_PARAMS= {
  "use_gpu": False,
  "replay_capacity": 500000,
  "batch_size": 32,
  "update_horizon": 1,
  "gamma": 0.99,
  "tau": 1.0,
  "lr": 2e-5,
  "loss_type": "huber",
  "train_frequency": 10,
  "target_update_frequency": 500,
  "save_model": False,
  "save_path": "",
  "seed": 42,
  "total_timesteps": 100000,
  "start_epsilon": 1.0,
  "end_epsilon": 0.05,
  "exploration_fraction": 0.2,
  "learning_starts": 1000
}
def load_parameters(param_file, env_name, agent_type):
    p = json.load(
        open(param_file, "r")
    )
    p["env_id"] = env_name
    p["agent"] = agent_type

    for key, value in DEFAULT_PARAMS.items():
        if key not in p.keys():
            p[key] = value

    return p