import gymnasium as gym
import numpy as np
import ale_py
from data_utils import ReplayMemory, parseEpisodes, save_model
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from Autoencoder import GaussianVAE
from transitionModel import transitionModel
import random

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the Atari Breakout environment
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")

memory = ReplayMemory(int(1e6))

# Observation transform
transform_width = 64
transform_height = 64
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((transform_height, transform_width)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ]
)

# Hyperparameters
train_interval = 100
batch_size = 32
transition_batch_size = 256
embedding_dim = 1024
lstm_input_size = 1024
hidden_size = 1024
seq_length = 4
num_layers = 4

# Models and optimizers
VAE = GaussianVAE(1, embedding_dim).to(device)
transitionModel = transitionModel(embedding_dim,lstm_input_size, hidden_size, num_layers, seq_length).to(device)
VAE_optim = torch.optim.AdamW(VAE.parameters(), lr=0.0002, betas=(0.9, 0.95), eps=1e-8)
transition_optim = torch.optim.AdamW(transitionModel.parameters(), lr=0.00002, betas=(0.9, 0.95), eps=1e-8)

torch.autograd.set_detect_anomaly(True)

num_episodes = 0
epochs=0
state_dicts = torch.load('./models/checkpoint_LSTM_TEST.pt',map_location=device)
VAE.load_state_dict(state_dicts['auto_encoder_state_dict'])
VAE.eval()
transitionModel.load_state_dict(state_dicts['transformer_model_state_dict'])
torch.compile(VAE)
torch.compile(transitionModel)

while True:  # Reset the environment to start a new episode
    obs, info = env.reset()
    obs = transform(obs).to(device)
    done = False
    total_reward = 0
    states, next_states, actions, rewards, dones = [], [], [], [], []

    while not done:
        # Randomly select an action from the action space
        action = env.action_space.sample()


        # Take the action in the environment
        observation, reward, done, truncated, info = env.step(action)
        next_state = transform(observation).to(device)

        # Accumulate the reward
        total_reward += reward

        # Append to episode data
        states.append(obs)
        next_states.append(next_state)
        actions.append(torch.tensor(action, device=device))
        rewards.append(torch.tensor(reward, device=device))
        dones.append(torch.tensor(done, device=device))
        obs = next_state

    # Push episode data to replay memory
    memory.push(states, actions, next_states, rewards, dones)

    num_episodes += 1
    if num_episodes % train_interval == 0:
        # Prepare data for training
        states, next_states, actions, rewards, dones, episode_lengths = parseEpisodes(memory)
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)

        shuffle = torch.randperm(states.shape[0], device=device)
        states = states[shuffle, :, :, :]
        states_batch = torch.split(states, batch_size, dim=0)

        # Train VAE
        vae_track = 0
        latents = []
        for batch in states_batch:
            VAE_optim.zero_grad()
            latent, recon, mu, var = VAE(batch)

            # recon_loss = nn.functional.mse_loss(batch, recon)
            # kl_divergence = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
            # loss = recon_loss# + .002*kl_divergence
            # vae_track += loss
            # loss.backward()
            # VAE_optim.step()
            latents.append(latent.detach())

        # # Unshuffle latents
        unshuffle = torch.argsort(shuffle)
        latents = torch.cat(latents, dim=0)
        latents = latents[unshuffle, :, :, :].squeeze()

        # Prepare conditioned latents
        conditioned_latents = torch.cat([latents, actions.unsqueeze(1)], dim=1)

        # Split into episodes
        #conditioned_latents = torch.split(conditioned_latents, episode_lengths, dim=0)
        rewards = torch.split(rewards, episode_lengths, dim=0)
        latents = torch.split(latents, episode_lengths, dim=0)
        actions = torch.split(actions, episode_lengths, dim=0)
        # Generate indices for training
        indices = [
            (i, j) for i in range(len(latents)) for j in range(len(latents[i]) - (seq_length + 1))
        ]
        random.shuffle(indices)

        # Train transition model
        num_batches = len(indices) // transition_batch_size
        transition_track = 0
        for batch_start in range(0, num_batches * transition_batch_size, transition_batch_size):
            batch_indices = indices[batch_start:batch_start + transition_batch_size]

            current_batch_input = torch.stack(
                [latents[ep][t:t + seq_length, :] for ep, t in batch_indices], dim=0
            ).to(device)

            current_batch_actions = torch.stack(
                [actions[ep][t:t+seq_length] for ep,t in batch_indices], dim=0
            ).to(device)

            current_batch_target = torch.stack(
                [latents[ep][t + 1, :] for ep, t in batch_indices], dim=0
            ).to(device)

            current_batch_reward = torch.stack(
                [rewards[ep][t + seq_length + 1] for ep, t in batch_indices], dim=0
            ).to(device)

            pred_latent, pred_reward = transitionModel(current_batch_input,current_batch_actions)

            transition_optim.zero_grad()
            latent_loss = nn.functional.mse_loss(current_batch_target, pred_latent)
            #reward_loss = nn.functional.mse_loss(current_batch_reward, pred_reward.squeeze(1))
            transition_loss = latent_loss# + reward_loss
            transition_track += latent_loss
            transition_loss.backward()
            print(transition_loss)
            #print("\nGradients for Transition Model:")
            # for name, param in transitionModel.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: {param.grad.norm().item()}")
            #     else:
            #         print(f"{name}: No gradient computed")
            transition_optim.step()
        epochs+=1
        save_model(VAE,transitionModel,VAE_optim,epochs,0,"./models/checkpoint_LSTM_TEST2.pt")
        #print("VAE Loss: ", vae_track / len(states_batch))

        print("Transition Loss: ", transition_track / num_batches)

    env.close()



