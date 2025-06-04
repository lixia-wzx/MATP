import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import math
from torch.autograd import Variable
from map_encoder import MapEncoder
from utils import get_dist, actor_gather, inverse
from spatial_temporal import ST
from PredNet import PredNet
from Attention_Block import AL_Attention_Block, LL_Attention_Block, LA_Attention_Block, AA_Attention_Block, Attention_Block_Edge


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # 创建位置编码矩阵
        pe = torch.zeros(self.max_seq_len, self.d_model).cuda()

        # 计算位置编码的值
        position = torch.arange(0, self.max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))

        # 调整位置编码矩阵的值
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 添加一个维度作为可学习的参数
        pe = pe.unsqueeze(0)  # (1,20,h)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将位置编码加到输入张量中
        # (B, Nax, 20, h)
        x = x + Variable(self.pe[:, :x.shape[2]], requires_grad=False)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        return self.mlp(x)


def get_dist_edge_attr(x1, x2):
    x1 = x1.unsqueeze(2)
    x2 = x2.unsqueeze(1)
    dist_edge_attr = x2 - x1  # (B,L,N,2)
    return dist_edge_attr


def get_masks(lane_num_nodes, agent_num_nodes, device):
    max_lane_num = max(lane_num_nodes)
    max_agent_num = max(agent_num_nodes)
    batch_size = len(lane_num_nodes)

    # === === Mask Generation Part === ===
    # === Agent - Agent Mask ===
    # query: agent, key-value: agent
    AA_mask = torch.zeros(
        batch_size, 1, 1, max_agent_num, max_agent_num, device=device)

    for i, agent_length in enumerate(agent_num_nodes):
        AA_mask[i, 0, 0, :agent_length, :agent_length] = 1
    # === === ===

    # === Agent - Lane Mask ===
    # query: agent, key-value: lane
    AL_mask = torch.zeros(
        batch_size, 1, 1, max_lane_num, max_agent_num, device=device)

    for i, (agent_length, lane_length) in enumerate(zip(agent_num_nodes, lane_num_nodes)):
        AL_mask[i, 0, 0, :lane_length, :agent_length] = 1
    # === === ===

    # === Lane - Lane Mask ===
    # query: lane, key-value: lane
    LL_mask = torch.zeros(
        batch_size, 1, 1, max_lane_num, max_lane_num, device=device)

    for i, lane_length in enumerate(lane_num_nodes):
        LL_mask[i, 0, 0, :lane_length, :lane_length] = 1

    # === === ===

    # === Lane - Agent Mask ===
    # query: lane, key-value: agent
    LA_mask = torch.zeros(
        batch_size, 1, 1, max_agent_num, max_lane_num, device=device)

    for i, (lane_length, agent_length) in enumerate(zip(lane_num_nodes, agent_num_nodes)):
        LA_mask[i, 0, 0, :agent_length, :lane_length] = 1

    # === === ===

    masks = [AL_mask, LL_mask, LA_mask, AA_mask]

    # === === === === ===

    return masks


# def get_adj_mask(data, num_historical_steps):
#     lane_true_position = pad_sequence(data["lane_true_position"], batch_first=True,
#                                       padding_value=torch.inf)  # (B,Lax,2)
#     lane_true_position = lane_true_position.unsqueeze(1).repeat(1, num_historical_steps, 1, 1)  # (B,20,Lax,2)
#     agent_true_position = pad_sequence(data["true_feats"], batch_first=True, padding_value=torch.inf)  # (B,Nax,20,2)
#     agent_true_position = agent_true_position.transpose(1, 2)  # (B,20,Nax,2)
#     # === === Mask Generation Part === ===
#     # === Agent - Agent Mask ===
#     AL_mask = torch.cdist(lane_true_position, agent_true_position, p=2)  # (B,20,Lax,Nax)
#     AL_mask = AL_mask <= 20.0
#     AL_mask = AL_mask.unsqueeze(2)  # (B,1,Lax,Nax)
#
#     LL_mask = torch.cdist(lane_true_position, lane_true_position, p=2)  # (B,20,Lax,Lax)
#     LL_mask = LL_mask <= 100.0
#     LL_mask = LL_mask.unsqueeze(2)  # (B,1,Lax,Lax)
#
#     LA_mask = torch.cdist(agent_true_position, lane_true_position, p=2)  # (B,20,Nax,Lax)
#     LA_mask = LA_mask <= 20.0
#     LA_mask = LA_mask.unsqueeze(2)  # (B,1,Nax,Lax)
#
#     AA_mask = torch.cdist(agent_true_position, agent_true_position, p=2)  # (B,20,Nax,Nax)
#     AA_mask = AA_mask <= 100.0
#     AA_mask = AA_mask.unsqueeze(2)  # (B,1,Nax,Nax)
#     masks = [AL_mask, LL_mask, LA_mask, AA_mask]
#     return masks


def get_adj_mask1(data):
    lane_true_position = pad_sequence(data["lane_position"], batch_first=True,
                                      padding_value=torch.inf)  # (B,Lax,2)
    agent_true_position = pad_sequence(data["ctrs"], batch_first=True, padding_value=torch.inf)  # (B,Nax,2)
    # === Agent - Agent Mask ===
    AL_mask = torch.cdist(lane_true_position, agent_true_position, p=2)  # (B,Lax,Nax)
    AL_mask = AL_mask <= 10.0
    AL_mask = AL_mask.unsqueeze(2).unsqueeze(3)  # (B,Lax,1,1,Nax)

    LL_mask = torch.cdist(lane_true_position, lane_true_position, p=2)  # (B,20,Lax,Lax)
    LL_mask = LL_mask <= 50.0
    LL_mask = LL_mask.unsqueeze(2).unsqueeze(3)  # (B,1,Lax,Lax)

    LA_mask = torch.cdist(agent_true_position, lane_true_position, p=2)  # (B,20,Nax,Lax)
    LA_mask = LA_mask <= 10.0
    LA_mask = LA_mask.unsqueeze(2).unsqueeze(3)  # (B,1,Nax,Lax)

    AA_mask = torch.cdist(agent_true_position, agent_true_position, p=2)  # (B,20,Nax,Nax)
    AA_mask = AA_mask <= 100.0
    AA_mask = AA_mask.unsqueeze(2).unsqueeze(3)  # (B,1,Nax,Nax)
    masks = [AL_mask, LL_mask, LA_mask, AA_mask]

    lane_true_position[lane_true_position == torch.inf] = 0
    agent_true_position[agent_true_position == torch.inf] = 0

    AL_dist = get_dist_edge_attr(lane_true_position, agent_true_position)  # (B,L,N,2)
    LL_dist = get_dist_edge_attr(lane_true_position, lane_true_position)  # (B,L,L,2)
    LA_dist = get_dist_edge_attr(agent_true_position, lane_true_position)  # (B,N,L,2)
    AA_dist = get_dist_edge_attr(agent_true_position, agent_true_position)  # (B,N,N,2)

    dist = [AL_dist, LL_dist, LA_dist, AA_dist]
    return masks, dist


class FusionNet(nn.Module):
    def __init__(self, hidden_size, heads, num_historical_steps, layer):
        super(FusionNet, self).__init__()
        self.layer = layer
        self.hidden_size = hidden_size
        self.num_historical_steps = num_historical_steps
        self.AL = nn.ModuleList([Attention_Block_Edge(hidden_size, heads) for _ in range(layer)])
        self.LL = nn.ModuleList([Attention_Block_Edge(hidden_size, heads) for _ in range(layer)])
        self.LA = nn.ModuleList([Attention_Block_Edge(hidden_size, heads) for _ in range(layer)])
        self.AA = nn.ModuleList([Attention_Block_Edge(hidden_size, heads) for _ in range(layer)])

    def forward(self, data, agent_features, lane_features):
        # (B,Nax,h),(B,Lax,h)
        # print("agent_features:",agent_features.shape)
        # print("lane_features:", lane_features.shape)
        mask, dist = get_adj_mask1(data)
        # lane_features = lane_features.unsqueeze(1).repeat(1, agent_features.shape[1], 1, 1)  # (B,T,Lax,h)
        for layer_index in range(self.layer):
            # === Agent to Lane ===
            lane_features = self.AL[layer_index](lane_features, agent_features, agent_features, mask[0], dist[0])
            # print("lane_features1:",lane_features.shape)
            lane_features = self.LL[layer_index](lane_features, lane_features, lane_features, mask[1], dist[1])
            # print("lane_features2:", lane_features.shape)
            agent_features = self.LA[layer_index](agent_features, lane_features, lane_features, mask[2], dist[2])
            # print("agent_features1:", agent_features.shape)
            agent_features = self.AA[layer_index](agent_features, agent_features, agent_features, mask[3], dist[3])
            # print("agent_features2:", agent_features.shape)
        return agent_features


class Model(nn.Module):
    def __init__(self, hidden_size, heads, layer, num_historical_steps, num_future_steps, device):
        super(Model, self).__init__()
        self.device = device
        self.num_historical_steps = num_historical_steps
        self.hidden_size = hidden_size
        self.pred = PredNet(hidden_size, num_historical_steps, num_future_steps)

        # self.fc_2 = MLP(2, hidden_size)
        # self.lstm = nn.GRU(3, hidden_size, batch_first=True)
        self.map = MapEncoder(hidden_dim=hidden_size, num_hops=4, num_heads=heads, dropout=0.1)
        self.fusion = FusionNet(hidden_size, heads, num_historical_steps, layer)
        self.st = ST(hidden_size, heads, num_historical_steps, layer, device)

    def forward(self, data):
        actor_idcs, my_actor_idcs, num_actors = actor_gather(data['feats'])
        edge_mask, att_mask = get_dist(data['true_feats'])  # (B,20,Nax,1,1,Nax)
        agent_features = pad_sequence(data["feats"], batch_first=True, padding_value=0).transpose(1, 2)  # (B,20,Nax,3)

        # batch_size = agent_features.shape[0]
        # agent_features = agent_features.reshape(-1, self.num_historical_steps, 3)  # (B*Nax,20,3)
        # agent_features = self.lstm(agent_features)[0]  # (B*Nax,20,h)
        # agent_features = agent_features.reshape(batch_size, -1, self.num_historical_steps,
        #                                           self.hidden_size).transpose(1, 2)  # (B,20,Nax,h)
        agent_features = self.st(agent_features, data["rot_feats"],edge_mask, att_mask)  # (B,Nax,h)

        lane_features = self.map(data)  # (B, Lax, h)
        agent_features = self.fusion(data, agent_features, lane_features)  # (B,Nax,h)

        # agent_features = agent_features.transpose(1, 2)  # (B,Nax,20,h)
        b = []
        for i in range(agent_features.shape[0]):
            b.append(agent_features[i, my_actor_idcs[i]])
        b = torch.cat(b, dim=0)  # (N*B, 20, h)

        last_out = self.pred(b, actor_idcs, data['ctrs'])
        rot, orig = data["rot"], data["orig"]
        last_out = inverse(last_out, data['ctrs'], rot, orig)
        return last_out
