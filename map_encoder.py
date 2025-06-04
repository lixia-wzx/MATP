import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttention
from layers import TwoLayerMLP
from utils import compute_angles_lengths_2D
from utils import init_weights
from utils import wrap_angle
from utils import compute_n_hop_edge_indices
from utils import transform_point_to_local_coordinate
from utils import generate_reachable_matrix
from torch.nn.utils.rnn import pad_sequence
from utils import transform_edge_index, transform_edge_index1, get_index


class MapEncoder(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 num_hops: int,
                 num_heads: int,
                 dropout: float) -> None:
        super(MapEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_hops = num_hops
        self.num_heads = num_heads
        self.dropout = dropout

        self._l2l_edge_type = ['adjacent', 'predecessor', 'successor']

        self.c_emb_layer = TwoLayerMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.l_emb_layer = TwoLayerMLP(input_dim=7, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.c2l_emb_layer = TwoLayerMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.l2l_emb_layer = TwoLayerMLP(input_dim=7, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.c2l_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout,
                                             has_edge_attr=True, if_self_attention=False)
        self.l2l_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout,
                                             has_edge_attr=True, if_self_attention=True)
        # self.meta = TwoLayerMLP(input_dim=hidden_dim+3, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.apply(init_weights)

    def forward(self, data):
        # embedding
        c_length = torch.cat(data["centerline_length"])  # (C_max)
        c2l_position_c = torch.cat(data["centerline_position"], dim=0)  # [(C1,...,Cb),2]  torch.Size([3069, 2])
        c2l_feats_c = torch.cat(data["centerline_feats"], dim=0)  # [(C1,...,Cb),2]  torch.Size([3069, 2])
        c2l_heading_c = torch.cat(data['centerline_heading'])  # [(C1,...,Cb)]  (3069)
        c_input = torch.cat([c2l_feats_c, c2l_heading_c.unsqueeze(-1), c_length.unsqueeze(-1)], dim=-1)
        c_embs = self.c_emb_layer(input=c_input)  # [(C1,...,Cb),D]

        c2l_position_l = torch.cat(data['lane_position'], dim=0)  # [(M1,...,Mb),2]   (341, 2)
        c2l_feats_l = torch.cat(data['lane_feats'], dim=0)  # [(M1,...,Mb),2]   (341, 2)
        l_length = torch.cat(data["lane_length"])  # (L_max)
        l_is_intersection = torch.cat(data["lane_is_intersection"]).unsqueeze(1)  # (3069, 1)
        l_turn_direction = torch.cat(data["lane_turn_direction"]).unsqueeze(1)  # (3069)
        l_traffic_control = torch.cat(data["lane_traffic_control"]).unsqueeze(1)  # (3069)
        c2l_heading_l = torch.cat(data['lane_heading'])  # [(M1,...,Mb)]  (341)
        l_input = torch.cat(
            [c2l_feats_l, l_length.unsqueeze(-1), c2l_heading_l.unsqueeze(-1), l_is_intersection, l_turn_direction,
             l_traffic_control],
            dim=-1)  #
        l_embs = self.l_emb_layer(input=l_input)  # [(M1,...,Mb),D]  # (341,64)
        # edge
        # c2l
        # print("c2l_position_c:", c2l_position_c.shape)
        # print("c2l_position_l:", c2l_position_l.shape)
        # print("c2l_heading_c:", c2l_heading_c.shape)
        # print("c2l_heading_l:", c2l_heading_l.shape)
        c2l_edge_index = transform_edge_index(data["centerline_to_lane_edge_index"], data["centerline_num_nodes"],
                                              data["lane_num_nodes"])
        # c2l_edge_index = torch.cat(data["centerline_to_lane_edge_index"], dim=1)  # [2,(C1,...,Cb)]  (2,3069)
        # print("c2l_edge_index:", c2l_edge_index.shape)
        c2l_edge_vector = transform_point_to_local_coordinate(c2l_position_c[c2l_edge_index[0]],
                                                              c2l_position_l[c2l_edge_index[1]],
                                                              c2l_heading_l[c2l_edge_index[1]])
        # print("c2l_edge_vector:", c2l_edge_vector.shape)   # (3069,2)
        c2l_edge_attr_length, c2l_edge_attr_theta = compute_angles_lengths_2D(c2l_edge_vector)
        c2l_edge_attr_heading = wrap_angle(c2l_heading_c[c2l_edge_index[0]] - c2l_heading_l[c2l_edge_index[1]])
        c2l_edge_attr_input = torch.stack([c2l_edge_attr_length, c2l_edge_attr_theta, c2l_edge_attr_heading], dim=-1)
        c2l_edge_attr_embs = self.c2l_emb_layer(input=c2l_edge_attr_input)  # (3069,64)
        # print("c2l_edge_attr_embs:", c2l_edge_attr_embs.shape)
        # l2l
        l2l_position = torch.cat(data['lane_position'], dim=0)  # [(M1,...,Mb),2]  (341,2)
        # print("l2l_position:", l2l_position.shape)
        l2l_heading = torch.cat(data['lane_heading'])  # [(M1,...,Mb)]  (341)
        # print("l2l_heading:", l2l_heading.shape)
        l2l_edge_index = []
        l2l_edge_attr_type = []
        l2l_edge_attr_hop = []

        # l2l_adjacent_edge_index = torch.cat(data['adjacent_edge_index'], dim=1)  # (2,234)
        l2l_adjacent_edge_index = transform_edge_index1(data['adjacent_edge_index'], data["lane_num_nodes"])
        num_adjacent_edges = l2l_adjacent_edge_index.size(1)
        l2l_edge_index.append(l2l_adjacent_edge_index)
        l2l_edge_attr_type.append(
            F.one_hot(torch.tensor(self._l2l_edge_type.index('adjacent')), num_classes=len(self._l2l_edge_type)).to(
                l2l_adjacent_edge_index.device).repeat(num_adjacent_edges, 1))
        l2l_edge_attr_hop.append(torch.ones(num_adjacent_edges, device=l2l_adjacent_edge_index.device))

        num_lanes = sum(data['lane_num_nodes'])
        # l2l_predecessor_edge_index = torch.cat(data['predecessor_edge_index'], dim=1)
        l2l_predecessor_edge_index = transform_edge_index1(data['predecessor_edge_index'], data["lane_num_nodes"])
        l2l_predecessor_edge_index_all = compute_n_hop_edge_indices(l2l_predecessor_edge_index, num_lanes,
                                                                    self.num_hops - 1,
                                                                    l2l_predecessor_edge_index.device)
        for i in range(self.num_hops):
            num_edges_now = l2l_predecessor_edge_index_all[i].size(1)
            l2l_edge_index.append(l2l_predecessor_edge_index_all[i])
            l2l_edge_attr_type.append(F.one_hot(torch.tensor(self._l2l_edge_type.index('predecessor')),
                                                num_classes=len(self._l2l_edge_type)).to(
                l2l_predecessor_edge_index.device).repeat(num_edges_now, 1))
            l2l_edge_attr_hop.append((i + 1) * torch.ones(num_edges_now, device=l2l_predecessor_edge_index.device))

        # l2l_successor_edge_index = torch.cat(data['successor_edge_index'], dim=1)
        l2l_successor_edge_index = transform_edge_index1(data['successor_edge_index'], data["lane_num_nodes"])
        l2l_successor_edge_index_all = compute_n_hop_edge_indices(l2l_successor_edge_index, num_lanes,
                                                                  self.num_hops - 1, l2l_successor_edge_index.device)
        for i in range(self.num_hops):
            num_edges_now = l2l_successor_edge_index_all[i].size(1)
            l2l_edge_index.append(l2l_successor_edge_index_all[i])
            l2l_edge_attr_type.append(F.one_hot(torch.tensor(self._l2l_edge_type.index('successor')),
                                                num_classes=len(self._l2l_edge_type)).to(
                l2l_successor_edge_index.device).repeat(num_edges_now, 1))
            l2l_edge_attr_hop.append((i + 1) * torch.ones(num_edges_now, device=l2l_successor_edge_index.device))

        l2l_edge_index = torch.cat(l2l_edge_index, dim=1)
        l2l_edge_attr_type = torch.cat(l2l_edge_attr_type, dim=0)
        l2l_edge_attr_hop = torch.cat(l2l_edge_attr_hop, dim=0)
        l2l_edge_vector = transform_point_to_local_coordinate(l2l_position[l2l_edge_index[0]],
                                                              l2l_position[l2l_edge_index[1]],
                                                              l2l_heading[l2l_edge_index[1]])
        l2l_edge_attr_length, l2l_edge_attr_theta = compute_angles_lengths_2D(l2l_edge_vector)
        l2l_edge_attr_heading = wrap_angle(l2l_heading[l2l_edge_index[0]] - l2l_heading[l2l_edge_index[1]])
        l2l_edge_attr_input = torch.cat(
            [l2l_edge_attr_length.unsqueeze(-1), l2l_edge_attr_theta.unsqueeze(-1), l2l_edge_attr_heading.unsqueeze(-1),
             l2l_edge_attr_hop.unsqueeze(-1), l2l_edge_attr_type], dim=-1)
        l2l_edge_attr_embs = self.l2l_emb_layer(input=l2l_edge_attr_input)  # torch.Size([3334, 64])
        # attention
        # c2l
        l_embs = self.c2l_attn_layer(x=[c_embs, l_embs], edge_index=c2l_edge_index,
                                     edge_attr=c2l_edge_attr_embs)  # [(M1,...,Mb),D]

        # l2l
        l_embs = self.l2l_attn_layer(x=l_embs, edge_index=l2l_edge_index,
                                     edge_attr=l2l_edge_attr_embs)  # [(M1,...,Mb),D]

        # lane_meta = torch.cat([l_is_intersection, l_traffic_control, l_turn_direction], dim=-1)
        # l_embs = self.meta(torch.cat((l_embs, lane_meta), 1))

        lane_idcs, lane_idcs_0_start = get_index(data['lane_num_nodes'])

        result = []
        for i in range(len(data['lane_num_nodes'])):
            result.append(l_embs[lane_idcs[i], :])
        result = pad_sequence(result, batch_first=True, padding_value=0)  # (B, Lax, h)

        return result
