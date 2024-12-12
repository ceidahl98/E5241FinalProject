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
import matplotlib.pyplot as plt

#torch.manual_seed(42)
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
transition_batch_size = 32
embedding_dim = 1024
hidden_size = 1024
seq_length = 4
num_layers = 4

# Models and optimizers
VAE = GaussianVAE(1, embedding_dim).to(device)
transitionModel = transitionModel(embedding_dim,1024, hidden_size, num_layers, seq_length).to(device)
VAE_optim = torch.optim.AdamW(VAE.parameters(), lr=0.0002, betas=(0.9, 0.95), eps=1e-8)
transition_optim = torch.optim.AdamW(transitionModel.parameters(), lr=0.0002, betas=(0.9, 0.95), eps=1e-8)

torch.autograd.set_detect_anomaly(True)
state_dicts = torch.load("./models/checkpoint_LSTM_TEST.pt",map_location=device)
VAE.load_state_dict(state_dicts['auto_encoder_state_dict'])
transitionModel.load_state_dict(state_dicts['transformer_model_state_dict'])
num_episodes = 0
epochs=0
VAE.eval()
transitionModel.eval()

obs_stack = []
actions_stack = []
while True:  # Reset the environment to start a new episode
    obs, info = env.reset()
    obs = transform(obs).to(device)

    done = False
    total_reward = 0
    states, next_states, actions, rewards, dones = [], [], [], [], []

    while not done:
        # Randomly select an action from the action space
        action = env.action_space.sample()

        z,test, _, _ = VAE(obs.unsqueeze(0))

        action = torch.tensor(action,device=device).view(1,1)
        #z = torch.cat([z.squeeze((2,3)),action],dim=1)
        actions_stack.append(action)
        obs_stack.append(z)
        if len(obs_stack) >= 5:
            fig = plt.figure(figsize=(8, 8))
            z = torch.cat(obs_stack,dim=0)
            print(z[-4:,:].unsqueeze(0).shape)
            actions_tensor = torch.cat(actions,dim=0)
            print(actions_tensor.shape)
            pred,_ = transitionModel(z[-4:,:].squeeze(),actions_tensor[-4:])
            print(pred.shape)
            pred = pred[-1,:].view(1,-1,1,1)
            print(pred.shape)
            image = VAE.decode(pred)
            print(image.shape)
            fig.add_subplot(2, 2, 1)
            plt.imshow(image.squeeze(0).permute(1, 2, 0).detach().numpy())

        # Take the action in the environment
        observation, reward, done, truncated, info = env.step(action)



        next_state = transform(observation).to(device)
        if len(obs_stack) >= 5:
            fig.add_subplot(2,2,2)
            plt.imshow(next_state.permute(1,2,0).detach().numpy())
            plt.show()

        # print(next_state.shape)
        # z, test, _, _ = VAE(next_state.unsqueeze(0))
        #
        # print(test.shape)
        # plt.imshow(test.squeeze(0).permute(1,2,0).detach().numpy())
        # plt.show()

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

            recon_loss = nn.functional.mse_loss(batch, recon)
            kl_divergence = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
            loss = recon_loss + kl_divergence
            vae_track += loss
            loss.backward()
            VAE_optim.step()
            latents.append(latent.detach())

        # # Unshuffle latents
        # unshuffle = torch.argsort(shuffle)
        # latents = torch.cat(latents, dim=0)
        # latents = latents[unshuffle, :, :, :].squeeze()
        #
        # # Prepare conditioned latents
        # conditioned_latents = torch.cat([latents, actions.unsqueeze(1)], dim=1)
        #
        # # Split into episodes
        # conditioned_latents = torch.split(conditioned_latents, episode_lengths, dim=0)
        # rewards = torch.split(rewards, episode_lengths, dim=0)
        # latents = torch.split(latents, episode_lengths, dim=0)
        #
        # # Generate indices for training
        # indices = [
        #     (i, j) for i in range(len(latents)) for j in range(len(latents[i]) - (seq_length + 1))
        # ]
        # random.shuffle(indices)
        #
        # # Train transition model
        # num_batches = len(indices) // transition_batch_size
        # transition_track = 0
        # for batch_start in range(0, num_batches * transition_batch_size, transition_batch_size):
        #     batch_indices = indices[batch_start:batch_start + transition_batch_size]
        #
        #     current_batch_input = torch.stack(
        #         [conditioned_latents[ep][t:t + seq_length, :] for ep, t in batch_indices], dim=0
        #     ).to(device)
        #
        #     current_batch_target = torch.stack(
        #         [latents[ep][t + 1:t + seq_length + 1, :] for ep, t in batch_indices], dim=0
        #     ).to(device)
        #
        #     current_batch_reward = torch.stack(
        #         [rewards[ep][t + seq_length + 1] for ep, t in batch_indices], dim=0
        #     ).to(device)
        #
        #     pred_latent, pred_reward = transitionModel(current_batch_input)
        #
        #     transition_optim.zero_grad()
        #     latent_loss = nn.functional.mse_loss(current_batch_target, pred_latent)
        #     reward_loss = nn.functional.mse_loss(current_batch_reward, pred_reward.squeeze(1))
        #     transition_loss = latent_loss + reward_loss
        #     transition_track += latent_loss
        #     transition_loss.backward()
        #     print("\nGradients for Transition Model:")
        #     # for name, param in transitionModel.named_parameters():
        #     #     if param.grad is not None:
        #     #         print(f"{name}: {param.grad.norm().item()}")
        #     #     else:
        #     #         print(f"{name}: No gradient computed")
        #     transition_optim.step()
        epochs+=1
        save_model(VAE,transitionModel,VAE_optim,epochs,vae_track.item() / len(states_batch),"./models/checkpoint.pt")
        print("VAE Loss: ", vae_track / len(states_batch))

        #print("Transition Loss: ", transition_track / num_batches)

    env.close()



