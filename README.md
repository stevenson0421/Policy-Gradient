# Policy-Gradient

Implementation of Policy Gradient

## Reinforce

- CartPole-v1

  - hidden size = 32
  - learning rate = 0.01
  - discounted factor = 0.999

  - average reward of last 100 episodes: 490.6

- Acrobot-v1

  - hidden size = 64
  - learning rate = 0.01
  - discounted factor = 0.999

  - average reward of last 100 episodes: -110.35

## Reinforce with Baseline

- CartPole-v1

  - hidden size = 32
  - learning rate = 0.001
  - discounted factor = 0.999
  - value loss ratio = 1

  - average reward of last 100 episodes: 494.44

- Acrobot-v1

  - hidden size = 32
  - learning rate = 0.01
  - discounted factor = 0.999
  - value loss ratio = 1

  - average reward of last 100 episodes: -91.27
