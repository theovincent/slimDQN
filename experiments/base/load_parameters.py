import json

PARAMS = [
  "use_gpu",
  "replay_capacity",
  "batch_size",
  "update_horizon",
  "gamma",
  "tau",
  "lr",
  "loss_type",
  "train_frequency",
  "target_update_frequency",
  "save_model",
  "save_path",
  "seed",
  "total_timesteps",
  "start_epsilon",
  "end_epsilon",
  "exploration_fraction",
  "learning_starts",
  "seed",
]

def load_parameters(param_file, env_name, agent_type, seed):
    p = json.load(
        open(param_file, "r")
    )
    p["env_id"] = env_name
    p["agent"] = agent_type
    p["seed"] = seed

    for key in PARAMS:
        if key not in p:
            AssertionError(f"Value for parameter {key} is not specified.")

    return p