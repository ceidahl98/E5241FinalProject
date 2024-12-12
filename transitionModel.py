import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class transitionModel(nn.Module):
    def __init__(self, input_size,embedding_size, hidden_size, num_layers, seq_length,dropout=0.2):
        super().__init__()
        self.seq_length=seq_length
        self.position_embedding = nn.Embedding(seq_length,input_size)
        self.action_embedding = nn.Embedding(4,embedding_size)
        self.input_embedding = nn.Linear(input_size,embedding_size)
        self.lstm = nn.LSTM(embedding_size,hidden_size,num_layers,batch_first=True,dropout=dropout)
        self.output_layer = nn.Linear(hidden_size,input_size,bias=False)
        self.reward_head = nn.Linear(input_size,1)

    def forward(self,x,action): #(B,S,L)
        print(x.shape)
        pos_emb = self.position_embedding(torch.arange(x.shape[1],device=device))
        input_emb = self.input_embedding(x)
        act_emb = self.action_embedding(action)

        emb = act_emb+input_emb+pos_emb

        x,hidden = self.lstm(emb)

        out = x[:,-1,:]
        reward = self.reward_head(out)
        #x = self.output_layer(reward_x)

        return out, reward
