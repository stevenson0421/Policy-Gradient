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


class ActorNetwork(torch.nn.Module):
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


class CriticNetwork(torch.nn.Module):
    def __init__(self, state_dimension, hidden_size=16):
        super().__init__()

        def init_parameters(layer, scale=1):
            torch.nn.init.xavier_normal_(layer.weight, gain=scale)
            torch.nn.init.zeros_(layer.bias)

        self.fc1 = torch.nn.Linear(in_features=state_dimension,
                                      out_features=hidden_size)
        init_parameters(self.fc1)
        self.fc2 = torch.nn.Linear(in_features=hidden_size,
                                      out_features=1)
        init_parameters(self.fc2)
        
        self.relu = torch.nn.ReLU()
        
    def forward(self, state):
        latent = self.relu(self.fc1(state))
        value = self.fc2(latent)

        return value


class ActorCriticNetwork(torch.nn.Module):
    def __init__(self, state_dimension, action_dimension, actor_learning_rate, critic_learning_rate, hidden_size):
        super().__init__()

        self.actor_network = ActorNetwork(state_dimension=state_dimension,
                                            action_dimension=action_dimension,
                                            hidden_size=hidden_size)
        
        self.critic_network = CriticNetwork(state_dimension=state_dimension,
                                  hidden_size=hidden_size)

        self.criterion = torch.nn.MSELoss()

        self.actor_optimizer = torch.optim.Adam(self.actor_network.parameters(),lr=actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic_network.parameters(),lr=critic_learning_rate)
        
    def step(self, state):
        policy = self.actor_network.forward(state)
        policy_distribution = torch.distributions.Categorical(probs=policy)
        action = policy_distribution.sample()

        self.action_log_probability = policy_distribution.log_prob(action)

        return action.cpu().item()
    
    def update(self, state, next_state, reward, discount_factor=0.999, device='cpu'):
        value = self.critic_network.forward(state)
        next_value = self.critic_network.forward(next_state)

        actor_loss = -self.action_log_probability * (reward + discount_factor * next_value - value).detach()
        critic_loss = self.criterion(reward + discount_factor * next_value, value)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss, critic_loss


def actor_critic(
    environment,
    target_score,
    networkclass,
    hidden_size,
    number_of_epoch,
    actor_learning_rate,
    critic_learning_rate,
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
                           actor_learning_rate=actor_learning_rate,
                           critic_learning_rate=critic_learning_rate,
                           hidden_size=hidden_size)

    network.to(device)

    start_time = time.time()
    average_trajectory_reward = deque(maxlen=100)

    step = 0

    for epoch in range(number_of_epoch):

        state, info = environment.reset(seed=epoch)

        state = torch.from_numpy(state).float().to(device).unsqueeze(0)

        trajectory_reward = 0
        trajectory_length = 0

        while True:
            action = network.step(state)

            next_state, reward, terminated, truncated, _ = environment.step(action)
            
            trajectory_reward += reward
            trajectory_length += 1
            
            next_state = torch.from_numpy(next_state).float().to(device).unsqueeze(0)

            actor_loss, critic_loss = network.update(state, next_state, reward, discount_factor, device)

            writer.add_scalar('Actor Loss', actor_loss, step)
            writer.add_scalar('Critic Loss', critic_loss, step)

            state = next_state
            step += 1

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
            break
    
    print(f'Train Time: {(time.time() - start_time):2f} seconds')
    print(f'Train Score: {np.mean(average_trajectory_reward)}')

    
    trajectory_reward = 0
    trajectory_length = 0
    screens = []
    state, info = environment.reset(seed=0)
    state = torch.from_numpy(state).float().to(device).unsqueeze(0)
    while True:
        with torch.no_grad():
            action = network.step(state)

        next_state, reward, terminated, truncated, info = environment.step(action)

        trajectory_reward += reward
        trajectory_length += 1

        screens.append(environment.render())

        state = torch.from_numpy(next_state).float().to(device).unsqueeze(0)

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
    torch.manual_seed(24)

    writer = SummaryWriter(log_dir=f'./runs/{game}_{time.strftime("%Y%m%d-%H%M%S")}')

    actor_critic(environment=gymnasium.make(game, render_mode='rgb_array'),
            target_score=120,
            networkclass=ActorCriticNetwork,
            hidden_size=64,
            number_of_epoch=10000,
            actor_learning_rate=0.0005,
            critic_learning_rate=0.0005,
            discount_factor=0.999,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            record_path=f'./video/{game}',
            record_name='actor_critic',
            record_frame=30,
            writer=writer)
        
    writer.close()

