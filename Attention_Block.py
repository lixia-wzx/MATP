import torch
from torch import nn
import torch.nn.functional as F


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


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self._linear1 = nn.Linear(d_model, d_ff)
        self._linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self._linear2(F.relu(self._linear1(x)))


class AL_Attention_Block(nn.Module):
    def __init__(self, hidden_size, heads):
        super(AL_Attention_Block, self).__init__()
        self.d_q = hidden_size // heads
        self.heads = heads
        self.hidden_size = hidden_size
        self.W_Q = nn.Linear(hidden_size, hidden_size)
        self.W_K = nn.Linear(hidden_size, hidden_size)
        self.W_V = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=0.2)
        self.feedForward = FeedForward(hidden_size, hidden_size * 2)

    def forward(self, input_Q, input_K, input_V, mask):
        # (B,T,Lax,h),(B,T,Nax,h),(B,T,Nax,h)
        residual = input_Q
        batch, max_num_agent, max_num_lane, hist_step = input_K.shape[0], input_K.shape[2], input_Q.shape[2], \
            input_K.shape[1]
        Q = self.W_Q(input_Q).reshape(batch, hist_step, max_num_lane, self.heads, self.d_q).permute(0, 1, 3, 2,
                                                                                                    4)  # (B, T, heads, L, d_v)
        K = self.W_K(input_K).reshape(batch, hist_step, max_num_agent, self.heads, self.d_q).permute(0, 1, 3, 2,
                                                                                                     4)  # (B, T, heads, N, d_v)
        V = self.W_V(input_V).reshape(batch, hist_step, max_num_agent, self.heads, self.d_q).permute(0, 1, 3, 2,
                                                                                                     4)  # (B, T, heads, N, d_v)
        attention = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_q ** 0.5)  # (B, T, heads, L, N)
        attention = attention.masked_fill_(mask == 0, value=-1e9)  # (B, T, heads, L, N)
        attention = F.softmax(attention, dim=-1)  # (B, T, heads, L, N)
        context = torch.matmul(attention, V)  # (B, T, heads, L, d_v)
        context = context.transpose(2, 3)  # (B, T, L, heads, d_v)
        context = context.reshape(batch, hist_step, max_num_lane, self.heads * self.d_q)  # (B, T, L, h)
        context = self.fc(context)  # (B, T, L, h)
        context = self.norm1(context + residual)  # (B, T, L, h)

        context_1 = self.feedForward(context)  # (B, T, L, h)
        last_out = self.norm2(context_1 + context)  # (B, T, L, h)
        return last_out


class LL_Attention_Block(nn.Module):
    def __init__(self, hidden_size, heads):
        super(LL_Attention_Block, self).__init__()
        self.d_q = hidden_size // heads
        self.heads = heads
        self.hidden_size = hidden_size
        self.W_Q = nn.Linear(hidden_size, hidden_size)
        self.W_K = nn.Linear(hidden_size, hidden_size)
        self.W_V = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=0.2)
        self.feedForward = FeedForward(hidden_size, hidden_size * 2)

    def forward(self, input_Q, input_K, input_V, mask):
        # (B,T,Lax,h)
        residual = input_Q
        batch, max_num_lane, hist_step = input_K.shape[0], input_Q.shape[2], input_Q.shape[1]
        Q = self.W_Q(input_Q).reshape(batch, hist_step, max_num_lane, self.heads, self.d_q).permute(0, 1, 3, 2,
                                                                                                    4)  # (B, T,heads, L, d_v)
        K = self.W_K(input_K).reshape(batch, hist_step, max_num_lane, self.heads, self.d_q).permute(0, 1, 3, 2,
                                                                                                    4)  # (B, T,heads, L, d_v)
        V = self.W_V(input_V).reshape(batch, hist_step, max_num_lane, self.heads, self.d_q).permute(0, 1, 3, 2,
                                                                                                    4)  # (B, T,heads, L, d_v)
        attention = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_q ** 0.5)  # (B, T,heads, L, L)
        attention = attention.masked_fill_(mask == 0, value=-1e9)  # (B, T,heads, L, L)
        attention = F.softmax(attention, dim=-1)  # (B, T, heads, L, L)
        context = torch.matmul(attention, V)  # (B, T,heads, L, d_v)
        context = context.transpose(2, 3)  # (B, T, L, heads, d_v)
        context = context.reshape(batch, hist_step, max_num_lane, self.heads * self.d_q)  # (B,T, L, h)
        context = self.fc(context)  # (B, T,L, h)
        context = self.norm1(context + residual)  # (B, T,L, h)

        context_1 = self.feedForward(context)  # (B,T, L, h)
        last_out = self.norm2(context_1 + context)  # (B,T, L, h)
        return last_out


class LA_Attention_Block(nn.Module):
    def __init__(self, hidden_size, heads):
        super(LA_Attention_Block, self).__init__()
        self.d_q = hidden_size // heads
        self.heads = heads
        self.hidden_size = hidden_size
        self.W_Q = nn.Linear(hidden_size, hidden_size)
        self.W_K = nn.Linear(hidden_size, hidden_size)
        self.W_V = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=0.2)
        self.feedForward = FeedForward(hidden_size, hidden_size * 2)

    def forward(self, input_Q, input_K, input_V, mask):
        # (B,T,Nax,h),(B,T,Lax,h),(B,T,Lax,h)
        residual = input_Q
        batch, max_num_agent, max_num_lane, hist_step = input_K.shape[0], input_Q.shape[2], input_K.shape[2], \
            input_Q.shape[1]
        Q = self.W_Q(input_Q).reshape(batch, hist_step, max_num_agent, self.heads, self.d_q).permute(0, 1, 3, 2,
                                                                                                     4)  # (B, T, heads, Nax, d_v)
        K = self.W_K(input_K).reshape(batch, hist_step, max_num_lane, self.heads, self.d_q).permute(0, 1, 3, 2,
                                                                                                    4)  # (B, T, heads, Nax, d_v)
        V = self.W_V(input_V).reshape(batch, hist_step, max_num_lane, self.heads, self.d_q).permute(0, 1, 3, 2,
                                                                                                    4)  # (B, T, heads, Nax, d_v)
        attention = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_q ** 0.5)  # (B, T, heads, Nax, Nax)
        attention = attention.masked_fill_(mask == 0, value=-1e9)  # (B, T, heads, Nax, Nax)
        attention = F.softmax(attention, dim=-1)  # (B, T, heads, Nax, Nax)
        context = torch.matmul(attention, V)  # (B, T, heads, Nax, d_v)
        context = context.transpose(2, 3)  # (B, T, Nax, heads, d_v)
        context = context.reshape(batch, hist_step, max_num_agent, self.heads * self.d_q)  # (B, T, Nax, h)
        context = self.fc(context)  # (B, T, Nax, h)
        context = self.norm1(context + residual)  # (B, T, Nax, h)

        context_1 = self.feedForward(context)  # (B, T, Nax, h)
        last_out = self.norm2(context_1 + context)  # (B, T, Nax, h)
        return last_out


class AA_Attention_Block(nn.Module):
    def __init__(self, hidden_size, heads):
        super(AA_Attention_Block, self).__init__()
        self.d_q = hidden_size // heads
        self.heads = heads
        self.hidden_size = hidden_size
        self.W_Q = nn.Linear(hidden_size, hidden_size)
        self.W_K = nn.Linear(hidden_size, hidden_size)
        self.W_V = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=0.2)
        self.feedForward = FeedForward(hidden_size, hidden_size * 2)

    def forward(self, input_Q, input_K, input_V, mask):
        # (B,T,Nax,h)
        residual = input_Q
        batch, max_num_agent, hist_step = input_K.shape[0], input_Q.shape[2], input_Q.shape[1]
        Q = self.W_Q(input_Q).reshape(batch, hist_step, max_num_agent, self.heads, self.d_q).permute(0, 1, 3, 2,
                                                                                                     4)  # (B, T, heads, Nax, d_v)
        K = self.W_K(input_K).reshape(batch, hist_step, max_num_agent, self.heads, self.d_q).permute(0, 1, 3, 2,
                                                                                                     4)  # (B,  T,heads, Nax, d_v)
        V = self.W_V(input_V).reshape(batch, hist_step, max_num_agent, self.heads, self.d_q).permute(0, 1, 3, 2,
                                                                                                     4)  # (B,  T,heads, Nax, d_v)
        attention = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_q ** 0.5)  # (B,  T,heads, Nax, Nax)
        attention = attention.masked_fill_(mask == 0, value=-1e9)  # (B,  T,heads, Nax, Nax)
        attention = F.softmax(attention, dim=-1)  # (B,  T,heads, Nax, Nax)
        context = torch.matmul(attention, V)  # (B, T, heads, Nax, d_v)
        context = context.transpose(2, 3)  # (B,  T, Nax, heads, d_v)
        context = context.reshape(batch, hist_step, max_num_agent, self.heads * self.d_q)  # (B, T, Nax, h)
        context = self.fc(context)  # (B, T, Nax, h)
        context = self.norm1(context + residual)  # (B, T, Nax, h)

        context_1 = self.feedForward(context)  # (B, T, Nax, h)
        last_out = self.norm2(context_1 + context)  # (B, T, Nax, h)
        return last_out


class Attention_Block(nn.Module):
    def __init__(self, hidden_size, heads):
        super(Attention_Block, self).__init__()
        self.d_q = hidden_size // heads
        self.heads = heads
        self.hidden_size = hidden_size
        self.W_Q = nn.Linear(hidden_size, hidden_size)
        self.W_K = nn.Linear(hidden_size, hidden_size)
        self.W_V = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=0.2)
        self.feedForward = FeedForward(hidden_size, hidden_size * 2)

    def forward(self, input_Q, input_K, input_V, mask):
        # AL:(B,Lax,h),(B,Nax,h),(B,Nax,h)
        # LL:(B,Lax,h),(B,Lax,h),(B,Lax,h)
        # LA:(B,Nax,h),(B,Lax,h),(B,Lax,h)
        # AA:(B,Nax,h),(B,Nax,h),(B,Nax,h)
        residual = input_Q
        batch, q_v, k_v = input_Q.shape[0], input_Q.shape[1], input_K.shape[1]
        Q = self.W_Q(input_Q).reshape(batch, q_v, self.heads, self.d_q).permute(0, 2, 1, 3)  # (B, heads, Nax, d_v)
        K = self.W_K(input_K).reshape(batch, k_v, self.heads, self.d_q).permute(0, 2, 1, 3)  # (B, heads, Nax, d_v)
        V = self.W_V(input_V).reshape(batch, k_v, self.heads, self.d_q).permute(0, 2, 1, 3)  # (B, heads, Nax, d_v)
        attention = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_q ** 0.5)  # (B, heads, Nax, Nax)
        attention = attention.masked_fill_(mask == 0, value=-1e9)  # (B, heads, Nax, Nax)
        attention = F.softmax(attention, dim=-1)  # (B, heads, Nax, Nax)
        context = torch.matmul(attention, V)  # (B, heads, Nax, d_v)
        context = context.transpose(1, 2)  # (B,  T, Nax, heads, d_v)
        context = context.reshape(batch, q_v, self.heads * self.d_q)  # (B, T, Nax, h)
        context = self.fc(context)  # (B, T, Nax, h)
        context = self.norm1(context + residual)  # (B, T, Nax, h)

        context_1 = self.feedForward(context)  # (B, T, Nax, h)
        last_out = self.norm2(context_1 + context)  # (B, T, Nax, h)
        return last_out


class Attention_Block_Edge(nn.Module):
    def __init__(self, hidden_size, heads):
        super(Attention_Block_Edge, self).__init__()
        self.d_q = hidden_size // heads
        self.heads = heads
        self.hidden_size = hidden_size
        self.W_Q = nn.Linear(hidden_size, hidden_size)
        self.W_K = nn.Linear(hidden_size, hidden_size)
        self.W_V = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.mlp = MLP(2, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.feedForward = FeedForward(hidden_size, hidden_size * 2)

    def forward(self, input_Q, input_K, input_V, mask, dist):
        # AL:(B,Lax,h),(B,Nax,h),(B,Nax,h), mask:(B,Lax,1,1,Nax), dist:(B,Lax,Nax,2)
        # LL:(B,Lax,h),(B,Lax,h),(B,Lax,h)
        # LA:(B,Nax,h),(B,Lax,h),(B,Lax,h)
        # AA:(B,Nax,h),(B,Nax,h),(B,Nax,h)

        input_K = input_K.unsqueeze(1) + self.mlp(dist)  # dist:(B,Lax,Nax,h)
        input_V = input_V.unsqueeze(1) + self.mlp(dist)
        # print("input_K:",input_K.shape)
        residual = input_Q
        batch, q_v, k_v = input_Q.shape[0], input_Q.shape[1], input_K.shape[2]
        Q = self.W_Q(input_Q).reshape(batch, q_v, 1, self.heads, self.d_q).permute(0, 1, 3, 2,
                                                                                   4)  # (B, Lax, heads, 1, d_v)
        K = self.W_K(input_K).reshape(batch, q_v, k_v, self.heads, self.d_q).permute(0, 1, 3, 2,
                                                                                     4)  # (B, Lax, heads, Nax, d_v)
        V = self.W_V(input_V).reshape(batch, q_v, k_v, self.heads, self.d_q).permute(0, 1, 3, 2,
                                                                                     4)  # (B, Lax, heads, Nax, d_v)
        attention = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_q ** 0.5)  # (B, Lax, heads, 1, Nax)
        # print(attention.shape)
        attention = attention.masked_fill_(mask == 0, value=-1e9)  # (B, Lax, heads, 1, Nax)
        attention = F.softmax(attention, dim=-1)  # (B, Lax, heads, 1, Nax)
        attention = self.dropout1(attention)
        context = torch.matmul(attention, V)  # (B, Lax, heads, 1, d_v)
        context = context.transpose(2, 3)  # (B, Lax, 1, heads, d_v)
        context = context.reshape(batch, q_v, self.heads * self.d_q)  # (B, Lax, h)
        context = self.fc(context)  # (B, Lax, h)
        context = self.dropout2(context)
        context = self.norm1(context + residual)  # (B, Lax, h)

        context_1 = self.feedForward(context)  # (B, Lax, h)
        last_out = self.norm2(context_1 + context)  # (B, Lax, h)
        return last_out
