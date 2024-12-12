from collections import deque, namedtuple
import random
import torch

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def pop_left(self):
        return self.memory.popleft()

    def __len__(self):
        return len(self.memory)


def parseEpisodes(memory):

    dataset = []
    actions = []
    states = []
    next_states = []
    rewards = []
    dones = []
    episode_lengths = []

    for _ in range(memory.__len__()):
        mem = memory.pop_left()
        #print(len(mem.state))
        episode_lengths.append(len(mem.state))
        dataset.append(mem)

    for episode in range(len(dataset)):

        states.append(torch.stack(dataset[episode].state))
        next_states.append(torch.stack(dataset[episode].next_state))
        rewards.append(torch.stack(dataset[episode].reward))
        actions.append(torch.stack(dataset[episode].action))
        dones.append(torch.stack(dataset[episode].done))
    #print(episode_lengths)
    states = torch.cat(states, dim=0)
    next_states = torch.cat(next_states,dim=0)
    actions = torch.cat(actions, dim=0)
    rewards = torch.cat(rewards, dim=0)
    dones = torch.cat(dones, dim=0)

    return states,next_states,actions,rewards,dones, episode_lengths


def save_model(auto_encoder, transformer_model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'auto_encoder_state_dict': auto_encoder.state_dict(),
        'transformer_model_state_dict': transformer_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Model saved to {path}")