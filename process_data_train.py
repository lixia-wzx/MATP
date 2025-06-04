from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
import torch
from scipy.spatial.distance import cdist
from tqdm import tqdm
import numpy as np
import copy
from scipy import sparse
from tqdm import tqdm
import warnings
from utils.process_data import compute_angles_lengths_2D, get_index_of_A_in_B

warnings.filterwarnings("ignore")

history_frames = 20
future_frames = 30
left_right = 1
pre_suc = 6

am = ArgoverseMap()


def process_data(idx, avl):
    name = avl.seq_list[idx].name[:-4]
    data = read_argo_data(avl[idx].seq_df, int(name))
    data = process_lane(data)
    torch.save(data, "/home/ME_4012_DATA2/xc1/wuzixuan/pre_train_data/" + name + ".pt")


def read_argo_data(df, name):
    city_name = df['CITY_NAME'].iloc[0]  # "MIA"
    seq_ts = np.sort(np.unique(df['TIMESTAMP'].values))

    mapping = dict()
    for i, ts in enumerate(seq_ts):
        mapping[ts] = i

    trajs = np.concatenate((
        df.X.to_numpy().reshape(-1, 1),
        df.Y.to_numpy().reshape(-1, 1)), 1)  # (337, 2)

    steps = [mapping[x] for x in df['TIMESTAMP'].values]
    steps = np.asarray(steps, np.int64)

    objs = df.groupby(['TRACK_ID', 'OBJECT_TYPE']).groups
    keys = list(objs.keys())

    obj_type = [x[1] for x in keys]

    agt_idx = obj_type.index('AGENT')  # 2
    idcs = objs[keys[agt_idx]]

    agt_traj = trajs[idcs]  # agent_features
    agt_step = steps[idcs]  # agent_step

    del keys[agt_idx]
    object_trajs, object_steps = [], []
    for key in keys:
        idcs = objs[key]
        object_trajs.append(trajs[idcs])
        object_steps.append(steps[idcs])

    data = dict()
    data["argo_id"] = name
    data['city'] = city_name
    data_trajs = [agt_traj] + object_trajs
    data_steps = [agt_step] + object_steps

    orig = data_trajs[0][19].copy().astype(np.float32)
    orig_1 = data_trajs[0][18].copy().astype(np.float32)
    pre = orig_1 - orig
    theta = np.pi - np.arctan2(pre[1], pre[0])
    rot = np.asarray([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]], np.float32)

    feats, ctrs, gt_preds, has_preds, true_feats, history_masks, rot_feats = [], [], [], [], [], [], []
    for traj, step in zip(data_trajs, data_steps):
        if 19 not in step:
            continue
        gt_pred = np.zeros((30, 2), np.float32)
        has_pred = np.zeros(30, np.bool)
        future_mask = np.logical_and(step >= 20, step < 50)
        post_step = step[future_mask] - 20
        post_traj = traj[future_mask]
        gt_pred[post_step] = post_traj
        has_pred[post_step] = 1

        obs_mask = step < 20
        step = step[obs_mask]
        traj = traj[obs_mask]
        idcs = step.argsort()
        step = step[idcs]
        traj = traj[idcs]

        for i in range(len(step)):
            if step[i] == 19 - (len(step) - 1) + i:
                break
        step = step[i:]
        traj = traj[i:]
        feat = np.zeros((20, 3), np.float32)
        true_feat = np.full((20,2), np.inf)
        # rot_feat = np.zeros((20, 2), np.float32)
        history_mask = np.zeros((20,), np.bool)
        history_mask[step] = 1
        true_feat[step] = traj
        feat[step, :2] = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T
        rot_feat = feat[:,:2].copy()
        feat[step, 2] = 1.0
        ctrs.append(feat[-1, :2].copy())
        feat[1:, :2] -= feat[:-1, :2]
        feat[step[0], :2] = 0
        feats.append(feat)
        true_feats.append(true_feat)
        rot_feats.append(rot_feat)
        gt_preds.append(gt_pred)
        has_preds.append(has_pred)
        history_masks.append(history_mask)

    feats = np.asarray(feats, np.float32)
    true_feats = np.asarray(true_feats, np.float32)
    rot_feats = np.asarray(rot_feats, np.float32)
    ctrs = np.asarray(ctrs, np.float32)
    gt_preds = np.asarray(gt_preds, np.float32)
    has_preds = np.asarray(has_preds, np.bool)
    history_masks = np.asarray(history_masks, np.bool)

    data['feats'] = torch.from_numpy(feats)
    data['true_feats'] = torch.from_numpy(true_feats)
    data['rot_feats'] = torch.from_numpy(rot_feats)
    data['ctrs'] = torch.from_numpy(ctrs)
    data['orig'] = torch.from_numpy(orig)
    data['theta'] = torch.tensor(theta)
    data['rot'] = torch.from_numpy(rot)
    data['gt_preds'] = torch.from_numpy(gt_preds)
    data['has_preds'] = torch.from_numpy(has_preds)
    data['history_masks'] = torch.from_numpy(history_masks)
    # data['dist_50_adj'] = get_dist_adj(data['true_feats'], 50)
    # data['dist_100_adj'] = get_dist_adj(data['true_feats'], 100)
    # data['exist_adj'] = get_exist_adj(data['history_masks'])  # (20,N,N)
    return data


def get_exist_adj(history_masks):
    history_masks = history_masks.transpose(0, 1)  # (20,N)
    exist_mask = torch.zeros(history_masks.shape[0], history_masks.shape[1], history_masks.shape[1])
    for t in range(history_masks.shape[0]):
        # 获取当前时刻的车辆存在情况
        current_time_vehicles = history_masks[t]
        # 找出当前时刻存在的车辆索引（值为1的列索引）
        exist_indices = torch.where(current_time_vehicles == 1)[0]
        # 对存在的车辆构建全连接矩阵
        for i in exist_indices:
            for j in exist_indices:
                exist_mask[t, i, j] = 1
    return exist_mask


def get_dist_adj(true_feats, radius):
    a = true_feats.transpose(0, 1)  # (20,N,2)
    dist_adj = torch.cdist(a, a, p=2)  # (20,N,N)
    distances = (dist_adj <= radius).to(torch.int64)
    # distances = distances.unsqueeze(0)  # (20,N,N)
    distances = get_lap(distances)
    distances = distances.unsqueeze(0)  # (1,20,N,N)
    return distances


def adjacency_matrix_to_degree_matrix(adjacency_matrix):
    degrees = adjacency_matrix.sum(dim=-1)
    degree_matrix = torch.zeros_like(adjacency_matrix)
    for i in range(adjacency_matrix.size(0)):
        degree_matrix[i] = torch.diag(degrees[i])
    degree_matrix = degree_matrix ** (-0.5)
    degree_matrix[degree_matrix == torch.inf] = 0
    return degree_matrix


def get_lap(adj):
    A_hat = adj + torch.eye(adj.shape[-1])
    dgree_matrix = adjacency_matrix_to_degree_matrix(A_hat)
    adj = torch.matmul(torch.matmul(dgree_matrix, A_hat), dgree_matrix)
    return adj


def process_lane(data):
    positions = data['true_feats'][:, :20][data['history_masks']].reshape(-1, 2)
    left_boundary = min(positions[:, 0])
    right_boundary = max(positions[:, 0])
    down_boundary = min(positions[:, 1])
    up_boundary = max(positions[:, 1])

    all_lane_ids = am.get_lane_ids_in_xy_bbox((left_boundary + right_boundary) / 2,
                                              (down_boundary + up_boundary) / 2, data['city'],
                                              max((right_boundary - left_boundary) / 2,
                                                  (up_boundary - down_boundary) / 2) + 50)
    # x_min, x_max, y_min, y_max = [-100.0, 100.0, -100.0, 100.0]
    # radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
    # all_lane_ids = am.get_lane_ids_in_xy_bbox(data['orig'][0], data['orig'][1], data['city'], radius)
    data["all_lane_ids"] = all_lane_ids

    lane_dict = {}
    for lane_id in all_lane_ids:
        lane = am.city_lane_centerlines_dict[data["city"]][lane_id]
        lane = copy.deepcopy(lane)
        centerline = torch.matmul(data['rot'],
                                  (torch.from_numpy(lane.centerline).float() - data['orig'].reshape(-1, 2)).T).T
        lane.centerline = centerline

        polygon = am.get_lane_segment_polygon(lane_id, data["city"])
        polygon = copy.deepcopy(polygon)
        lane.polygon = torch.matmul(data['rot'],
                                    (torch.from_numpy(polygon[:, :2]).float() - data['orig'].reshape(-1, 2)).T).T
        lane_dict[lane_id] = lane
    lane_ids = list(lane_dict.keys())

    num_lanes = len(lane_ids)
    lane_position = torch.zeros(num_lanes, 2, dtype=torch.float)
    # lane_true_position = torch.zeros(num_lanes, 2, dtype=torch.float)
    lane_heading = torch.zeros(num_lanes, dtype=torch.float)
    lane_length = torch.zeros(num_lanes, dtype=torch.float)
    lane_is_intersection = torch.zeros(num_lanes, dtype=torch.uint8)
    lane_turn_direction = torch.zeros(num_lanes, dtype=torch.uint8)
    lane_traffic_control = torch.zeros(num_lanes, dtype=torch.uint8)

    num_centerlines = torch.zeros(num_lanes, dtype=torch.long)
    centerline_position = [None] * num_lanes
    centerline_heading = [None] * num_lanes
    centerline_length = [None] * num_lanes
    centerline_feats = []
    lane_feats = []

    lane_adjacent_edge_index = []
    lane_predecessor_edge_index = []
    lane_successor_edge_index = []
    for lane_id in lane_ids:
        lane_idx = lane_ids.index(lane_id)
        centerlines = lane_dict[lane_id].centerline
        num_centerlines[lane_idx] = centerlines.size(0) - 1
        centerline_position[lane_idx] = (centerlines[1:] + centerlines[:-1]) / 2
        centerline_feat = centerlines[1:] - centerlines[:-1]
        centerline_feats.append(centerline_feat)

        centerline_length[lane_idx], centerline_heading[lane_idx] = compute_angles_lengths_2D(centerline_feat)
        lane_length[lane_idx] = centerline_length[lane_idx].sum()

        center_index = int(centerline_position[lane_idx].size(0) / 2)
        # lane_position[lane_idx] = centerlines[center_index]
        lane_position[lane_idx] = centerline_position[lane_idx][center_index]
        lane_feats.append(centerline_position[lane_idx][-1:] - centerline_position[lane_idx][:1])
        # lane_true_position[lane_idx] = torch.from_numpy(
        #     am.city_lane_centerlines_dict[data["city"]][lane_id].centerline[center_index])
        lane_heading[lane_idx] = torch.atan2(centerlines[center_index + 1, 1] - centerlines[center_index, 1],
                                             centerlines[center_index + 1, 0] - centerlines[center_index, 0])

        lane_is_intersection[lane_idx] = torch.tensor(am.lane_is_in_intersection(lane_id, data['city']),
                                                      dtype=torch.uint8)
        lane_turn_direction[lane_idx] = torch.tensor(
            ['NONE', 'LEFT', 'RIGHT'].index(lane_dict[lane_id].turn_direction), dtype=torch.uint8)
        lane_traffic_control[lane_idx] = torch.tensor(lane_dict[lane_id].has_traffic_control,
                                                      dtype=torch.uint8)

        lane_adjacent_ids = am.get_lane_segment_adjacent_ids(lane_id, data['city'])
        lane_adjacent_idx = get_index_of_A_in_B(lane_adjacent_ids, lane_ids)
        if len(lane_adjacent_idx) != 0:
            edge_index = torch.stack([torch.tensor(lane_adjacent_idx, dtype=torch.long),
                                      torch.full((len(lane_adjacent_idx),), lane_idx, dtype=torch.long)], dim=0)
            lane_adjacent_edge_index.append(edge_index)
        lane_predecessor_ids = am.get_lane_segment_predecessor_ids(lane_id, data['city'])
        lane_predecessor_idx = get_index_of_A_in_B(lane_predecessor_ids, lane_ids)
        if len(lane_predecessor_idx) != 0:
            edge_index = torch.stack([torch.tensor(lane_predecessor_idx, dtype=torch.long),
                                      torch.full((len(lane_predecessor_idx),), lane_idx, dtype=torch.long)], dim=0)
            lane_predecessor_edge_index.append(edge_index)
        lane_successor_ids = am.get_lane_segment_successor_ids(lane_id, data['city'])
        lane_successor_idx = get_index_of_A_in_B(lane_successor_ids, lane_ids)
        if len(lane_successor_idx) != 0:
            edge_index = torch.stack([torch.tensor(lane_successor_idx, dtype=torch.long),
                                      torch.full((len(lane_successor_idx),), lane_idx, dtype=torch.long)], dim=0)
            lane_successor_edge_index.append(edge_index)

    data['lane_num_nodes'] = num_lanes
    data['lane_position'] = lane_position
    data['lane_feats'] = torch.cat(lane_feats, dim=0)
    # data['lane_true_position'] = lane_true_position
    data['lane_length'] = lane_length
    data['lane_heading'] = lane_heading
    data['lane_is_intersection'] = lane_is_intersection
    data['lane_turn_direction'] = lane_turn_direction
    data['lane_traffic_control'] = lane_traffic_control

    data['centerline_num_nodes'] = num_centerlines.sum().item()
    data['centerline_position'] = torch.cat(centerline_position, dim=0)
    data['centerline_feats'] = torch.cat(centerline_feats, dim=0)
    data['centerline_heading'] = torch.cat(centerline_heading, dim=0)
    data['centerline_length'] = torch.cat(centerline_length, dim=0)

    centerline_to_lane_edge_index = torch.stack([torch.arange(num_centerlines.sum(), dtype=torch.long),
                                                 torch.arange(num_lanes, dtype=torch.long).repeat_interleave(
                                                     num_centerlines)], dim=0)
    data['centerline_to_lane_edge_index'] = centerline_to_lane_edge_index

    if len(lane_adjacent_edge_index) != 0:
        lane_adjacent_edge_index = torch.cat(lane_adjacent_edge_index, dim=1)
    else:
        lane_adjacent_edge_index = torch.tensor([[], []], dtype=torch.long)
    lane_predecessor_edge_index = torch.cat(lane_predecessor_edge_index, dim=1)
    lane_successor_edge_index = torch.cat(lane_successor_edge_index, dim=1)

    data['adjacent_edge_index'] = lane_adjacent_edge_index
    data['predecessor_edge_index'] = lane_predecessor_edge_index
    data['successor_edge_index'] = lane_successor_edge_index

    return data


if __name__ == "__main__":
    avl = ArgoverseForecastingLoader(
        r"/home/ME_4012_DATA2/xc1/wuzixuan/wargo_data/train/data")
    for i in tqdm(range(205942)):
        process_data(i, avl)
