import gymnasium

import numpy as np
import scipy
import time
import os
import cv2
from tqdm import trange
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter


def discounted_cumulated_sum(sequence, discount_factor, device=None):
    if isinstance(sequence, np.ndarray):
        return scipy.signal.lfilter([1], [1, float(-discount_factor)], sequence[::-1], axis=0)[::-1]
    elif isinstance(sequence, torch.Tensor):
        return torch.as_tensor(np.ascontiguousarray(scipy.signal.lfilter([1], [1, float(-discount_factor)], sequence.detach().cpu().numpy()[::-1], axis=0)[::-1]), dtype=torch.float32, device=device)
    else:
        raise TypeError


class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dimension, action_dimension, hidden_size=16):
        super().__init__()

        def init_parameters(layer, scale=1):
            torch.nn.init.xavier_normal_(layer.weight, gain=scale)
            torch.nn.init.zeros_(layer.bias)

        self.state_dimension = state_dimension

        self.share1 = torch.nn.Linear(in_features=state_dimension,
                                      out_features=hidden_size)
        init_parameters(self.share1)
        self.mean1 = torch.nn.Linear(in_features=hidden_size,
                                      out_features=action_dimension)
        init_parameters(self.mean1)

        self.std1 = torch.nn.Linear(in_features=hidden_size,
                                      out_features=action_dimension)
        init_parameters(self.std1)
        
        self.activation = torch.nn.Tanh()

        self.action_log_probs = []
        self.rewards = []

        self.log_std = torch.nn.Parameter(torch.from_numpy(-0.5*np.ones(action_dimension, dtype=np.float32)))
        
    def forward(self, state):
        latent = self.activation(self.share1(state))
        mean = self.activation(self.mean1(latent))
        log_std = self.activation(self.std1(latent))

        return mean, log_std
    
    def step(self, state):
        mean, log_std = self.forward(state)
        policy_distribution = torch.distributions.Normal(mean, log_std.exp())
        action = policy_distribution.sample()
        action_log_probability = policy_distribution.log_prob(action).sum(axis=1)

        self.action_log_probs.append(action_log_probability)

        return action.view(-1).cpu().numpy()
    
    def get_loss(self, discount_factor=0.999, device='cpu'):
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device='cpu')
        reward_to_go = discounted_cumulated_sum(rewards, discount_factor, device)
        reward_to_go = (reward_to_go - reward_to_go.mean()) / reward_to_go.std()

        # policy loss = sum of log pi(action_t|state_t) * G_t over t
        policy_losses = [-a * b for a, b in zip(self.action_log_probs, reward_to_go)]
        loss = torch.cat(policy_losses).sum()

        return loss
    
    def clear_memory(self):
        # reset rewards and action buffer
        del self.action_log_probs[:]
        del self.rewards[:]


def reinforce(environment,
              networkclass,
              hidden_size,
              number_of_epoch,
              learning_rate,
              discount_factor,
              device,
              record_path,
              record_name,
              record_frame,
              writer):

    state_dimension = environment.observation_space.shape[0]
    print('Observation Dimension: ', state_dimension)
    action_dimension = environment.action_space.shape[0]
    print('Action Dimension: ', action_dimension)
    action_scale = (environment.action_space.high - environment.action_space.low) / 2.0

    network = networkclass(state_dimension=state_dimension,
                           action_dimension=action_dimension,
                           hidden_size=hidden_size)

    network.to(device)
    
    optimizer = torch.optim.Adam(network.parameters(), learning_rate)

    def update(epoch):
        loss = network.get_loss(discount_factor, device)

        writer.add_scalar('Policy Loss', loss, epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    def test():
        trajectory_reward = 0
        trajectory_length = 0
        screens = []
        state, info = environment.reset(seed=0)
        while True:
            with torch.no_grad():
                action = network.step(torch.from_numpy(state).to(device).unsqueeze(0))
                action *= action_scale
            next_state, reward, terminated, truncated, info = environment.step(action)

            trajectory_reward += reward
            trajectory_length += 1

            screens.append(environment.render())

            state = next_state

            if (terminated or truncated):
                print(f'trajectory ends with length {trajectory_length}: reward: {trajectory_reward}')

                if not os.path.exists(record_path):
                    os.makedirs(record_path)
                out = cv2.VideoWriter(os.path.join(record_path, f'{record_name}.avi'),cv2.VideoWriter_fourcc(*'DIVX'), record_frame, (screens[0].shape[1], screens[0].shape[0]))
                for img in screens:
                    out.write(img)
                out.release()
                
                trajectory_reward = 0
                trajectory_length = 0

                break

    start_time = time.time()
    average_trajectory_reward = deque(maxlen=100)

    for epoch in range(number_of_epoch):

        state, info = environment.reset(seed=0)

        trajectory_reward = 0
        trajectory_length = 0

        while True:
            action = network.step(torch.from_numpy(state).to(device).unsqueeze(0))
            action *= action_scale

            next_state, reward, terminated, truncated, _ = environment.step(action)

            trajectory_reward += reward
            trajectory_length += 1

            # buffer.store(action_log_probability, reward)
            network.rewards.append(reward)
            
            state = next_state

            if terminated or truncated:
                average_trajectory_reward.append(trajectory_reward)
                writer.add_scalar('Trajectory Reward', trajectory_reward, epoch)
                writer.add_scalar('Average Trajectory Reward', np.mean(average_trajectory_reward), epoch)
                print(f'trajectory ends with reward: {trajectory_reward}, length: {trajectory_length}')

                trajectory_reward = 0
                trajectory_length = 0
                
                break

        update(epoch)

        network.clear_memory()

    writer.close()
    
    print(f'Train Time: {(time.time() - start_time):2f} seconds')
    print(f'Train Score: {np.mean(average_trajectory_reward)}')

    test()

    environment.close()


if __name__ == '__main__':
    game = 'MountainCarContinuous-v0'
    if not os.path.exists('./runs/'):
        os.makedirs('./runs/')
    writer = SummaryWriter(log_dir=f'./runs/{game}_{time.strftime("%Y%m%d-%H%M%S")}')
    torch.manual_seed(24)
    reinforce(environment=gymnasium.make(game, render_mode='rgb_array'),
            networkclass=PolicyNetwork,
            hidden_size=32,
            number_of_epoch=1000,
            learning_rate=0.001,
            discount_factor=0.999,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            record_path=f'./video/{game}',
            record_name='reinforce',
            record_frame=30,
            writer=writer)
