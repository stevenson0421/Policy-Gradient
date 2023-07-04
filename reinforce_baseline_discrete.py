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
        return torch.from_numpy(np.ascontiguousarray(scipy.signal.lfilter([1], [1, float(-discount_factor)], sequence.detach().cpu().numpy()[::-1], axis=0)[::-1])).to(device)
    else:
        raise TypeError


class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dimension, action_dimension, hidden_size=16):
        super().__init__()

        def init_parameters(layer, scale=1):
            torch.nn.init.xavier_normal_(layer.weight, gain=scale)
            torch.nn.init.zeros_(layer.bias)

        self.state_dimension = state_dimension

        self.fc1 = torch.nn.Linear(in_features=state_dimension,
                                      out_features=hidden_size)
        init_parameters(self.fc1)
        self.fc2 = torch.nn.Linear(in_features=hidden_size,
                                      out_features=action_dimension)
        init_parameters(self.fc2)
        
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)
        
    def forward(self, state):
        latent = self.relu(self.fc1(state))
        policy = self.softmax(self.fc2(latent))

        return policy


class ValueNetwork(torch.nn.Module):
    def __init__(self, state_dimension, hidden_size=16):
        super().__init__()

        def init_parameters(layer, scale=1):
            torch.nn.init.xavier_normal_(layer.weight, gain=scale)
            torch.nn.init.zeros_(layer.bias)

        self.state_dimension = state_dimension

        self.fc1 = torch.nn.Linear(in_features=state_dimension,
                                      out_features=hidden_size)
        init_parameters(self.fc1)
        self.fc2 = torch.nn.Linear(in_features=hidden_size,
                                      out_features=1)
        init_parameters(self.fc2)
        
        self.relu = torch.nn.ReLU()

        self.values = []
        
    def forward(self, state):
        latent = self.relu(self.fc1(state))
        value = self.fc2(latent)

        return value
    
    

class PolicyValueNetwork(torch.nn.Module):
    def __init__(self, state_dimension, action_dimension, hidden_size=16):
        super().__init__()

        self.policy_network = PolicyNetwork(state_dimension=state_dimension,
                                            action_dimension=action_dimension,
                                            hidden_size=hidden_size)
        
        self.value_network = ValueNetwork(state_dimension=state_dimension,
                                          hidden_size=hidden_size)
        
        self.action_log_probs = []
        self.rewards = []
        self.values = []

        self.criterion = torch.nn.MSELoss()
        
    def step(self, state):
        policy = self.policy_network.forward(state)
        policy_distribution = torch.distributions.Categorical(probs=policy)
        action = policy_distribution.sample()
        action_log_probability = policy_distribution.log_prob(action)

        self.action_log_probs.append(action_log_probability)

        value = self.value_network.forward(state)

        self.values.append(value)

        return action.cpu().item()
    
    def get_loss(self, discount_factor=0.999, device='cpu'):
        rewards = torch.as_tensor(self.rewards, dtype=torch.float32, device='cpu')
        reward_to_go = discounted_cumulated_sum(rewards, discount_factor, device)

        values = torch.cat(self.values).squeeze().double()

        advantages = reward_to_go - values

        # policy loss = sum of log pi(action_t|state_t) * (G_t - V_t) over t
        policy_losses = [-a * b for a, b in zip(self.action_log_probs, advantages)]
        policy_loss = torch.cat(policy_losses).sum()
        
        # value loss = 1/|T| * (G_t - V_t)^2
        value_loss = self.criterion(values, reward_to_go)

        return policy_loss, value_loss
    
    def clear_memory(self):
        del self.action_log_probs[:]
        del self.rewards[:]
        del self.values[:]


def reinforce_baseline(
    environment,
    target_score,
    networkclass,
    hidden_size,
    number_of_epoch,
    learning_rate,
    value_loss_ratio,
    discount_factor,
    device,
    record_path,
    record_name,
    record_frame,
    writer):

    state_dimension = environment.observation_space.shape[0]
    action_dimension = environment.action_space.n

    network = networkclass(state_dimension=state_dimension,
                           action_dimension=action_dimension,
                           hidden_size=hidden_size)

    network.to(device)
    
    optimizer = torch.optim.Adam(network.parameters(), learning_rate)

    start_time = time.time()
    average_trajectory_reward = deque(maxlen=100)

    for epoch in range(number_of_epoch):

        state, info = environment.reset(seed=epoch)

        state = torch.from_numpy(state).to(device).unsqueeze(0)

        trajectory_reward = 0
        trajectory_length = 0

        while True:
            action = network.step(state)

            next_state, reward, terminated, truncated, _ = environment.step(action)

            trajectory_reward += reward
            trajectory_length += 1

            # buffer.store(action_log_probability, reward)
            network.rewards.append(reward)
            
            state = torch.from_numpy(next_state).to(device).unsqueeze(0)

            if terminated or truncated:
                average_trajectory_reward.append(trajectory_reward)
                writer.add_scalar('Trajectory Reward', trajectory_reward, epoch)
                writer.add_scalar('Average Trajectory Reward', np.mean(average_trajectory_reward), epoch)
                print(f'trajectory ends with reward: {trajectory_reward}, length: {trajectory_length}')

                trajectory_reward = 0
                trajectory_length = 0
                
                break
    
        if np.mean(average_trajectory_reward) >= target_score:
            print(f'solved with {epoch} epochs')
            network.clear_memory()
            break

        policy_loss, value_loss = network.get_loss(discount_factor, device)
        total_loss = policy_loss + value_loss_ratio * value_loss

        writer.add_scalar('Policy Loss', policy_loss, epoch)
        writer.add_scalar('Value Loss', value_loss, epoch)
        writer.add_scalar('Total Loss', total_loss, epoch)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        network.clear_memory()

    writer.close()
    
    print(f'Train Time: {(time.time() - start_time):2f} seconds')
    print(f'Train Score: {np.mean(average_trajectory_reward)}')

    trajectory_reward = 0
    trajectory_length = 0
    screens = []
    state, info = environment.reset(seed=0)
    state = torch.from_numpy(state).to(device).unsqueeze(0)
    while True:
        with torch.no_grad():
            action = network.step(state)

        next_state, reward, terminated, truncated, info = environment.step(action)

        trajectory_reward += reward
        trajectory_length += 1

        screens.append(environment.render())

        state = torch.from_numpy(next_state).to(device).unsqueeze(0)

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

    environment.close()


if __name__ == '__main__':
    game = 'LunarLander-v2'
    if not os.path.exists('./runs/'):
        os.makedirs('./runs/')
    writer = SummaryWriter(log_dir=f'./runs/{game}_{time.strftime("%Y%m%d-%H%M%S")}')
    torch.manual_seed(24)
    reinforce_baseline(environment=gymnasium.make(game, render_mode='rgb_array'),
            target_score=120,
            networkclass=PolicyValueNetwork,
            hidden_size=128,
            number_of_epoch=10000,
            learning_rate=0.001,
            value_loss_ratio=1,
            discount_factor=0.999,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            record_path=f'./video/{game}',
            record_name='reinforce_baseline',
            record_frame=30,
            writer=writer)

