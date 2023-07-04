# Policy-Gradient

Implementation of Policy Gradient

## Reinforce

- CartPole-v1

  - random seed = 24
  - hidden size = 64
  - learning rate = 0.01
  - discounted factor = 0.999

  - average reward of last 100 episodes reach 500 in 480 epochs

- Acrobot-v1

  - random seed = 24
  - hidden size = 64
  - learning rate = 0.01
  - discounted factor = 0.999

  - average reward of last 100 episodes reach -100 in 674 epochs

- LunarLander-v2

  - random seed = 24
  - hidden size = 64
  - learning rate = 0.0005
  - discounted factor = 0.999

  - average reward of last 100 episodes reach 120 in 8076 epochs

## Reinforce with Baseline

- CartPole-v1

  - random seed = 24
  - hidden size = 32
  - learning rate = 0.005
  - discounted factor = 0.999
  - value loss ratio = 1

  - average reward of last 100 episodes reach 500 in 734 epochs

- Acrobot-v1

  - random seed = 24
  - hidden size = 32
  - learning rate = 0.005
  - discounted factor = 0.999
  - value loss ratio = 1

  - average reward of last 100 episodes reach -100 in 619 epochs

- LunarLander-v2

  - random seed = 24
  - hidden size = 64
  - learning rate = 0.005
  - discounted factor = 0.999

  - average reward of last 100 episodes reach 120 in 4892 epochs


## Advantage Actor Critic

- CartPole-v1

  - random seed = 24
  - hidden size = 64
  - learning rate = 0.0005 / 0.0005
  - discounted factor = 0.999

  - average reward of last 100 episodes reach 500 in 521 epochs

- Acrobot-v1

  - random seed = 24
  - hidden size = 64
  - learning rate = 0.0005 / 0.0005
  - discounted factor = 0.999

  - average reward of last 100 episodes reach -100 in 257 epochs

- LunarLander-v2

  - random seed = 24
  - hidden size = 64
  - learning rate = 0.0005
  - discounted factor = 0.999

  - average reward of last 100 episodes reach 120 in 601 epochs
