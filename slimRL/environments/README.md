Add your custom environment into this folder with the following structure:

```python
class Env:
    def __init__(self, ...):
        # environment variables here

    def step(self, action):
        # step function here for the given action

    def reset(self):
        # reset env to a random state
}
```

For running DQN algorithm on the environment, add:

```python
class EnvDQN(Env):
    def __init__(self, ...):
        super().__init__(self, ...)

    def abc(self, ...):
        # additional functions you want to add in your DQN run for the environment
```