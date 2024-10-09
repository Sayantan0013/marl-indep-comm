import torch.nn as nn
import torch.nn.functional as F
from numpy import sqrt

import torch

'''
Because the RNN is used here, the last hidden_state is required each time. For an episode of data, each obs needs the last hidden_state to select the action.
Therefore, it is not possible to directly and randomly extract a batch of experience input to the neural network, so a batch of episodes is needed here, and the transition of the same position of this batch of episodes is passed in each time.
In this case, the hidden_state can be saved, and the next experience is the next experience
'''

class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args
        self.attention = Attention(input_shape + args.rnn_hidden_dim, args)
        self.fc1 = nn.Linear(input_shape + args.final_msg_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)  # rnn gating mechanism; gated recurrent unit
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.input_shape = input_shape
        self.msg_dim = args.final_msg_dim

    def forward(self, obs, hidden_state, msgs=None, agent_num=None):
        # if communicating        
        if self.args.with_comm:

            ep_num = 1
            if agent_num == None:
                ep_num = obs.shape[0] // self.args.n_agents
            
            msgs_rec = msgs

            h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
            q = self.attention(obs,h_in)

            # select the messages only from the other agetns, i.e., remove the ones of agent_num: [n_agents - 1, obs_dim]
            if agent_num != None:
                # Removing its own message
                # idxs = torch.tensor([i for i in range(self.args.n_agents) if i!=agent_num]).to("cuda" if self.args.cuda else "cpu")
                # msgs_rec = torch.index_select(msgs_rec, dim=1, index=idxs)
                # separating key value pair
                keys, values = msgs_rec[:,:,:self.args.key_dim], msgs_rec[:,:,self.args.key_dim:]
                # calculting attention vector and final comm
                alpha = F.softmax(keys@q.T/sqrt(self.args.key_dim),dim=-1)
                agg_msg = (values.transpose(1,2)@alpha).transpose(1,2)
                obs = torch.cat((obs, agg_msg.reshape(obs.shape[0], -1)), dim=-1)

            else:
                # during training everything comes together (bs >= 1), so need another way to cat the respective messages to the right indices

                msgs_rec = msgs_rec.repeat(1, self.args.n_agents, 1).reshape(ep_num, self.args.n_agents, self.args.n_agents, -1)

                keys, values = msgs_rec[:,:,:,:self.args.key_dim], msgs_rec[:,:,:,self.args.key_dim:]
                q = q.reshape(ep_num,self.args.n_agents,1,-1)
                # calculting attention vector and final comm
                alpha = F.softmax((keys*q).sum(dim=-1)/sqrt(self.args.key_dim),dim=-1).unsqueeze(-1)
                agg_msg = (values*alpha).sum(-2)

                # cat messages to the inputs to the policy network
                # obs here is in shape [bs * n_a, input_dim]; need to change to [bs, n_a, input_dim]
                obs_aux = obs.reshape(ep_num, self.args.n_agents, -1)
                # now concat with msgs_rec and change to previous shape: [bs, n_a, input_dim+msg_dim] -> [bs*n_a, input_dim+msg_dim]
                obs = torch.cat((obs_aux, agg_msg), dim=-1).reshape(ep_num * self.args.n_agents, -1)


        ## Plan ##
        ## We will first send the observation with last hidden state to generate a query ##
        ## This query vector will then be used to generate the attention weight for all the agents comms ##
        ## Then we will follow the same scheme but with less dims ##

        x = F.relu(self.fc1(obs))
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        return q, h

class Attention(nn.Module):
    def __init__(self, input_shape, args):
        super(Attention, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.attention_hidden_dim)
        self.fc2 = nn.Linear(args.attention_hidden_dim, args.key_dim)
    
    def forward(self, obs, rnn_hidden):
        x = torch.cat([obs,rnn_hidden],dim=-1)
        x = F.relu(self.fc1(x))
        q = self.fc2(x)
        return q

# Critic of Central-V
class Critic(nn.Module):
    def __init__(self, input_shape, args):
        super(Critic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.critic_dim)
        self.fc2 = nn.Linear(args.critic_dim, args.critic_dim)
        self.fc3 = nn.Linear(args.critic_dim, 1)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q
