import torch
from torch import nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, n_in, n_out, act=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(n_out)
        self.act = act

    def forward(self, x):
        # (B,N,h)
        x = self.linear(x)
        x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x


class LinearRes(nn.Module):
    def __init__(self, n_in, n_out):
        super(LinearRes, self).__init__()

        self.linear1 = nn.Linear(n_in, n_out, bias=False)
        self.linear2 = nn.Linear(n_out, n_out, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = nn.LayerNorm(n_out)
        self.norm2 = nn.LayerNorm(n_out)

        if n_in != n_out:
            self.transform = nn.Sequential(
                nn.Linear(n_in, n_out, bias=False),
                nn.LayerNorm(n_out))
        else:
            self.transform = None

    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.norm2(out)

        if self.transform is not None:
            out += self.transform(x)
        else:
            out += x

        out = self.relu(out)
        return out


class AttDest(nn.Module):
    def __init__(self, n_agt):
        super(AttDest, self).__init__()
        self.dist = nn.Sequential(
            nn.Linear(2, n_agt),
            nn.ReLU(),
            Linear(n_agt, n_agt),
        )

        self.agt = Linear(2 * n_agt, n_agt)

    def forward(self, agts, agt_ctrs, dest_ctrs):
        # (B*N, 128),  (B*N, 2), (B*N, 6, 2)
        # print("agts:",agts.shape)
        # print("agt_ctrs:", agt_ctrs.shape)
        # print("dest_ctrs:", dest_ctrs.shape)
        n_agt = agts.size(1)
        num_mods = dest_ctrs.size(1)

        dist = (agt_ctrs.unsqueeze(1) - dest_ctrs).reshape(-1, 2)  # (B*N*6,2)
        dist = self.dist(dist)  # (B*N*6,128)
        agts = agts.unsqueeze(1).repeat(1, num_mods, 1).reshape(-1, n_agt)  # (B*N*6,128)

        agts = torch.cat((dist, agts), 1)  # (B*N*6,256)
        agts = self.agt(agts)  # (B*N*6,128)
        return agts


class PredNet(nn.Module):
    """
    Final motion forecasting with Linear Residual block
    """

    def __init__(self, hidden_size, num_historical_steps, num_future_steps):
        super(PredNet, self).__init__()
        # n_actor = hidden_size  # 128

        pred = []
        for i in range(6):  # 6
            pred.append(
                nn.Sequential(
                    LinearRes(hidden_size, hidden_size),
                    nn.Linear(hidden_size, 2 * num_future_steps),
                )
            )
        self.pred = nn.ModuleList(pred)
        self.att_dest = AttDest(hidden_size)
        self.cls = nn.Sequential(
            LinearRes(hidden_size, hidden_size), nn.Linear(hidden_size, 1)
        )

    def forward(self, actors, actor_idcs, actor_ctrs):
        # (B*N, 128)
        # agents = actors.reshape(actors.shape[0], -1)  # (B*N, 10*128)
        preds = []
        for layer in self.pred:
            preds.append(layer(actors))  # (B*N,60)
        reg = torch.cat([x.unsqueeze(1) for x in preds], 1)  # (B*N,6,60)

        reg = reg.view(reg.size(0), reg.size(1), -1, 2)

        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 2)  # (N,1,1,2)
            reg[idcs] = reg[idcs] + ctrs

        dest_ctrs = reg[:, :, -1].detach()  # (B*N, 6, 2)
        feats = self.att_dest(actors, torch.cat(actor_ctrs, 0), dest_ctrs)  # (B*N*6,128)
        cls = self.cls(feats).view(-1, 6)  # (B*N,6)
        # cls = F.softmax(cls, dim=-1)
        cls, sort_idcs = cls.sort(1, descending=True)
        row_idcs = torch.arange(len(sort_idcs)).long().to(sort_idcs.device)
        row_idcs = row_idcs.view(-1, 1).repeat(1, sort_idcs.size(1)).view(-1)  # (B*N*6)
        sort_idcs = sort_idcs.view(-1)  # (B*N*6)
        reg = reg[row_idcs, sort_idcs].view(cls.size(0), cls.size(1), -1, 2)  # (B*N,6,30,2)
        out = dict()
        out["cls"], out["reg"] = [], []
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            out["cls"].append(cls[idcs])  # (N1,6)
            out["reg"].append(reg[idcs])  # (N1,6,30,2)
        return out
