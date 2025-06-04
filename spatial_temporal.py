import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pad_sequence


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


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self._linear1 = nn.Linear(d_model, d_ff)
        self._linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self._linear2(F.relu(self._linear1(x)))


class TemporalAtt(nn.Module):
    def __init__(self, hidden_size, heads, num_historical_steps):
        super(TemporalAtt, self).__init__()
        self.d_q = hidden_size // heads
        self.heads = heads
        self.hidden_size = hidden_size
        self.num_historical_steps = num_historical_steps
        self.W_Q = nn.Linear(hidden_size, hidden_size)
        self.W_K = nn.Linear(hidden_size, hidden_size)
        self.W_V = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.feedForward = FeedForward(hidden_size, hidden_size * 2)

    def forward(self, input_Q, input_K, input_V):
        # (B,Nax,20,h),
        residual = input_Q
        batch, max_num_object = input_Q.shape[0], input_Q.shape[1]
        Q = self.W_Q(input_Q).reshape(batch, max_num_object, self.num_historical_steps, self.heads, self.d_q).permute(0,
                                                                                                                      1,
                                                                                                                      3,
                                                                                                                      2,
                                                                                                                      4)  # (B,Nax, heads, 20, d_v)
        K = self.W_K(input_K).reshape(batch, max_num_object, self.num_historical_steps, self.heads, self.d_q).permute(0,
                                                                                                                      1,
                                                                                                                      3,
                                                                                                                      2,
                                                                                                                      4)  # (B,Nax, heads, 20, d_v)
        V = self.W_V(input_V).reshape(batch, max_num_object, self.num_historical_steps, self.heads, self.d_q).permute(0,
                                                                                                                      1,
                                                                                                                      3,
                                                                                                                      2,
                                                                                                                      4)  # (B,Nax, heads, 20, d_v)
        attention = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_q ** 0.5)  # (B, Nax, heads, 20, 20)
        # attention = attention.masked_fill_(temporal_mask == 0, value=-1e9)  # (B, Nax, heads, 20, 20)
        attention = F.softmax(attention, dim=-1)  # (B, Nax, heads, 20, 20)
        attention = self.dropout1(attention)
        context = torch.matmul(attention, V)  # (B, Nax, heads, 20, d_v)
        context = context.transpose(2, 3)  # (B, Nax, 20, heads, d_v)
        context = context.reshape(batch, max_num_object, self.num_historical_steps,
                                  self.heads * self.d_q)  # (B, Nax, 20, h)
        context = self.fc(context)  # (B, 20, Nax, h)
        context = self.dropout2(context)
        context = self.norm1(context + residual)  # (B, Nax, 20, h)

        context_1 = self.feedForward(context)  # (B, Nax, 20, h)
        last_out = self.norm2(context_1 + context)  # (B, Nax, 20, h)
        return last_out


class SpatialAtt(nn.Module):
    def __init__(self, hidden_size, heads, num_historical_steps):
        super(SpatialAtt, self).__init__()
        self.d_q = hidden_size // heads
        self.heads = heads
        self.hidden_size = hidden_size
        self.num_historical_steps = num_historical_steps
        self.W_Q = nn.Linear(hidden_size, hidden_size)
        self.W_K = nn.Linear(hidden_size, hidden_size)
        self.W_V = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.feedForward = FeedForward(hidden_size, hidden_size * 2)

    def forward(self, input_Q, input_K, input_V, distances_mask):
        # (B,20,Nax,h),
        residual = input_Q
        batch, max_num_object = input_Q.shape[0], input_Q.shape[2]
        Q = self.W_Q(input_Q).reshape(batch, self.num_historical_steps, max_num_object, self.heads, self.d_q).permute(0,
                                                                                                                      1,
                                                                                                                      3,
                                                                                                                      2,
                                                                                                                      4)  # (B,Nax, heads, 20, d_v)
        K = self.W_K(input_K).reshape(batch, self.num_historical_steps, max_num_object, self.heads, self.d_q).permute(0,
                                                                                                                      1,
                                                                                                                      3,
                                                                                                                      2,
                                                                                                                      4)  # (B,Nax, heads, 20, d_v)
        V = self.W_V(input_V).reshape(batch, self.num_historical_steps, max_num_object, self.heads, self.d_q).permute(0,
                                                                                                                      1,
                                                                                                                      3,
                                                                                                                      2,
                                                                                                                      4)  # (B,Nax, heads, 20, d_v)
        attention = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_q ** 0.5)  # (B, Nax, heads, 20, 20)
        attention = attention.masked_fill_(distances_mask == 0, value=-1e9)  # (B, Nax, heads, 20, 20)
        attention = F.softmax(attention, dim=-1)  # (B, Nax, heads, 20, 20)
        attention = self.dropout1(attention)
        context = torch.matmul(attention, V)  # (B, Nax, heads, 20, d_v)
        context = context.transpose(2, 3)  # (B, Nax, 20, heads, d_v)
        context = context.reshape(batch, self.num_historical_steps, max_num_object,
                                  self.heads * self.d_q)  # (B, Nax, 20, h)
        context = self.fc(context)  # (B, 20, Nax, h)
        context = self.dropout2(context)
        context = self.norm1(context + residual)  # (B, Nax, 20, h)

        context_1 = self.feedForward(context)  # (B, Nax, 20, h)
        last_out = self.norm2(context_1 + context)  # (B, Nax, 20, h)
        return last_out


class SpatialAttEdge(nn.Module):
    def __init__(self, hidden_size, heads, num_historical_steps):
        super(SpatialAttEdge, self).__init__()
        self.d_q = hidden_size // heads
        self.heads = heads
        self.hidden_size = hidden_size
        self.num_historical_steps = num_historical_steps
        self.W_Q = nn.Linear(hidden_size, hidden_size)
        self.W_K = nn.Linear(hidden_size, hidden_size)
        self.W_V = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(p=0.1)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(p=0.1)
        self.feedForward = FeedForward(hidden_size, hidden_size * 2)

    def forward(self, input_Q, input_K, input_V, distances_mask):
        # (B,20,N,1,3), (B,20,N,N,5), (B,20,N,N,5)
        # print("input_Q:", input_Q.shape)
        residual = input_Q
        input_Q = input_Q.unsqueeze(3)
        batch, max_num_object = input_Q.shape[0], input_Q.shape[2]
        Q = self.W_Q(input_Q).reshape(batch, self.num_historical_steps, max_num_object, 1, self.heads,
                                      self.d_q).permute(0, 1, 2, 4, 3, 5)  # (B, 20, Nax, heads, 1, d_v)
        K = self.W_K(input_K).reshape(batch, self.num_historical_steps, max_num_object, max_num_object, self.heads,
                                      self.d_q).permute(0, 1, 2, 4, 3, 5)  # (B, 20, Nax, heads, Nax, d_v)
        V = self.W_V(input_V).reshape(batch, self.num_historical_steps, max_num_object, max_num_object, self.heads,
                                      self.d_q).permute(0, 1, 2, 4, 3, 5)  # (B, 20, Nax, heads, Nax, d_v)
        attention = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_q ** 0.5)  # (B, 20, Nax, heads, 1, Nax)
        # print("attention:", attention.shape)
        # print("distances_mask:", distances_mask.shape)
        attention = attention.masked_fill_(distances_mask == 0, value=-1e9)  # (B, 20, Nax, heads, 1, Nax)
        attention = F.softmax(attention, dim=-1)  # (B, 20, Nax, heads, 1, Nax)
        attention = self.dropout1(attention)
        context = torch.matmul(attention, V)  # (B, 20, Nax, heads, 1, d_v)
        context = context.transpose(3, 4)  # (B, 20, Nax, 1, heads, d_v)
        context = context.reshape(batch, self.num_historical_steps, max_num_object,
                                  self.heads * self.d_q)  # (B, 20, Nax, h)
        context = self.fc(context)  # (B, 20, Nax, h)
        context = self.dropout2(context)
        context = self.norm1(context + residual)  # (B, Nax, 20, h)

        context_1 = self.feedForward(context)  # (B, Nax, 20, h)
        last_out = self.norm2(context_1 + context)  # (B, Nax, 20, h)
        return last_out


class GCN(nn.Module):
    def __init__(self, hidden_size, layer=3):
        super(GCN, self).__init__()
        self.layer = layer
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, dist_adj):
        # x:(B,20,Nax,h)
        # print("x:",x.shape)
        # print("dist_adj:", dist_adj.shape)
        x = self.fc(torch.matmul(dist_adj, x))
        return x


class Spatial(nn.Module):
    def __init__(self, hidden_size, heads, num_historical_steps):
        super(Spatial, self).__init__()
        self.satt = SpatialAtt(hidden_size, heads, num_historical_steps)
        # self.gcn = GCN(hidden_size)
        # self.fc1 = MLP(hidden_size, hidden_size)
        # self.fc2 = MLP(hidden_size, hidden_size)

    def forward(self, agent_features, distances_mask):
        satt_out = self.satt(agent_features, agent_features, agent_features, distances_mask)
        # gcn_out = self.gcn(object_features, dist_adj)
        #
        # z = torch.sigmoid(self.fc1(gcn_out) + self.fc2(satt_out))
        # last_out = z * gcn_out + (1 - z) * satt_out
        return satt_out


class Temporal(nn.Module):
    def __init__(self, hidden_size, heads, num_historical_steps):
        super(Temporal, self).__init__()
        self.tatt = TemporalAtt(hidden_size, heads, num_historical_steps)
        # self.tcn = TCN(hidden_size, [3, 6, 12], [1, 2, 4], 4, 0.1)
        # self.fc1 = MLP(hidden_size, hidden_size)
        # self.fc2 = MLP(hidden_size, hidden_size)

    def forward(self, agent_features):
        tatt_out = self.tatt(agent_features, agent_features, agent_features)
        # tcn_out = self.tcn(object_features)

        # z = torch.sigmoid(self.fc1(tcn_out) + self.fc2(tatt_out))
        # last_out = z * tatt_out + (1 - z) * tcn_out
        return tatt_out


def get_spatial_edge(rot_feats, object_features):
    rot_feats = pad_sequence(rot_feats, batch_first=True, padding_value=0)  # (B,Nax,20,2)
    rot_feats = rot_feats.transpose(1, 2)  # (B,20,Nax,2)
    edge = rot_feats.unsqueeze(2) - rot_feats.unsqueeze(3)  # (B,20,Nax,Nax,2)
    edge = torch.cat([edge, object_features.unsqueeze(2).repeat(1, 1, edge.shape[2], 1, 1)], dim=-1)  # (B,20,Nax,Nax,5)
    # print("edge:", edge.shape)
    return edge


class ST(nn.Module):
    def __init__(self, hidden_size, heads, num_historical_steps, layer, device):
        super(ST, self).__init__()
        self.layer = layer
        self.num_historical_steps = num_historical_steps
        self.spatial_att_edge = SpatialAttEdge(hidden_size, heads, num_historical_steps)
        self.spatial_att = nn.ModuleList([SpatialAtt(hidden_size, heads, num_historical_steps) for _ in range(layer-1)])
        self.temporal = nn.ModuleList([TemporalAtt(hidden_size, heads, num_historical_steps) for _ in range(layer)])
        self.fc1 = MLP(3, hidden_size)
        self.fc2 = MLP(5, hidden_size)
        self.fc3 = MLP(hidden_size, hidden_size)
        self.pe = PositionalEncoding(hidden_size, 20)
        self.device = device

    def forward(self, agent_features, rot_feats, edge_mask, att_mask):
        # (B,20,Nax,2)
        kv_spatial = get_spatial_edge(rot_feats, agent_features)  # (B,20,Nax,Nax,5)
        agent_features = self.fc1(agent_features)  # (B,20,Nax,h)
        kv_spatial = self.fc2(kv_spatial)
        # out = [agent_features]
        for layer in range(self.layer):
            if layer == 0:
                agent_features = self.spatial_att_edge(agent_features, kv_spatial, kv_spatial,
                                                       edge_mask)  # (B,20,Nax,h)
            else:
                agent_features = self.spatial_att[layer - 1](agent_features, agent_features, agent_features,
                                                             att_mask)
            # agent_features = agent_features.transpose(1, 2)  # (B,Nax,20,h)
            # agent_features = self.pe(agent_features)
            # agent_features = self.temporal[layer](agent_features, agent_features, agent_features)  # (B,Nax,20,h)
            # agent_features = agent_features.transpose(1, 2)  # (B,20,Nax,h)
            # out.append(agent_features)
        # a = 0
        # for i in out:
        #     a += i
        # a = self.fc3(a)
        return agent_features[:, -1]  # (B,Nax,h)
