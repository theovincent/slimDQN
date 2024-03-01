All environments mandatorily need the following functions to be implemented:
```python
class Env:
    def __init__(self, ...):
        # environment variables here

    def step(self, action):
        # step function here for the given action

    def reset(self):
        # reset env to a random state
```
Sufficient functions needed to run various algorithms on specific environments:
## 1. CarOnHill

###DQN

Above definition is sufficient

## 2. ChainWalk

###DQN

Above definition is sufficient