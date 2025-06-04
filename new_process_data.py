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


def get_distance(xy1, xy2):
    dis = np.sqrt(np.sum((xy1 - xy2) ** 2))
    return dis


# 获得前驱、后继、左右邻居的pairs
def get_pairs(city_name, nearby_lane_ids, lane_dict):
    # pre_pairs = []
    # suc_pairs = []
    left_pairs = []
    right_pairs = []
    for lane_id in nearby_lane_ids:
        lane = lane_dict[lane_id]

        # if lane.predecessors is not None:
        #     for nbr_id in lane.predecessors:
        #         if nbr_id in nearby_lane_ids:
        #             pre_pairs.append([lane_id, nbr_id])

        # if lane.successors is not None:
        #     for nbr_id in lane.successors:
        #         if nbr_id in nearby_lane_ids:
        #             suc_pairs.append([lane_id, nbr_id])

        nbr_id = lane.l_neighbor_id
        if nbr_id is not None:
            if nbr_id in nearby_lane_ids:
                left_pairs.append([lane_id, nbr_id])

        nbr_id = lane.r_neighbor_id
        if nbr_id is not None:
            if nbr_id in nearby_lane_ids:
                right_pairs.append([lane_id, nbr_id])
    # 去重
    # pre_list = []
    # suc_list = []
    left_list = []
    right_list = []
    # for x in pre_pairs:
    #     if x not in pre_list:
    #         pre_list.append(x)

    # for x in suc_pairs:
    #     if x not in suc_list:
    #         suc_list.append(x)

    for x in left_pairs:
        if x not in left_list:
            left_list.append(x)

    for x in right_pairs:
        if x not in right_list:
            right_list.append(x)
    return left_pairs, right_pairs


def get_lap(adj):
    D = np.sum(adj, axis=1)
    D = np.power(D, -0.5)
    D[D == np.inf] = 0
    D = np.diag(D)
    lap = np.matmul(np.matmul(D, adj), D)
    return sparse.csr_matrix(lap)


def get_adj_m(adj_pairs, m, num_nodes):
    adj_list = []
    adj_list.append(adj_pairs)
    adj = sparse.csr_matrix((np.ones(len(adj_pairs[0])), (adj_pairs[0], adj_pairs[1])), shape=(num_nodes, num_nodes))
    mat = adj
    for i in range(1, m):
        adj = adj.dot(mat)
        adj_list.append(np.array(adj.nonzero()))
    return adj_list


def get_adj(data, adj_pairs, num_nodes):
    # pre_adj = sparse.csr_matrix((np.ones(len(adj_pairs[0][0])), (adj_pairs[0][1], adj_pairs[0][0])),
    #                             shape=(num_nodes, num_nodes))
    # suc_adj = sparse.csr_matrix((np.ones(len(adj_pairs[1][0])), (adj_pairs[1][1], adj_pairs[1][0])),
    #                             shape=(num_nodes, num_nodes))
    # left_adj = sparse.csr_matrix((np.ones(len(adj_pairs[2][0])), (adj_pairs[2][1], adj_pairs[2][0])),
    #                              shape=(num_nodes, num_nodes))
    # right_adj = sparse.csr_matrix((np.ones(len(adj_pairs[3][0])), (adj_pairs[3][1], adj_pairs[3][0])),
    #                               shape=(num_nodes, num_nodes))

    data["pre_adj_list"] = get_adj_m(adj_pairs[0], 6, num_nodes)
    data["suc_adj_list"] = get_adj_m(adj_pairs[1], 6, num_nodes)
    data["left_adj_list"] = get_adj_m(adj_pairs[2], 1, num_nodes)
    data["right_adj_list"] = get_adj_m(adj_pairs[3], 1, num_nodes)
    return data


def process_data(idx, avl):
    name = avl.seq_list[idx].name[:-4]
    data, data_trajs, data_steps = read_argo_data(avl[idx].seq_df, int(name))
    data = process_lane(data, data_trajs, data_steps)
    torch.save(data, "/home/ME_4012_DATA2/xc1/wuzixuan/my_train_data/" + name + ".pt")


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

    origin_objects_xy, features_x, ctrs, features_y, has_preds = [], [], [], [], []
    for traj, step in zip(data_trajs, data_steps):
        if 19 not in step:
            continue
        feature_y = np.zeros((future_frames, 2), np.float32)
        has_pred = np.zeros(future_frames, bool)
        future_mask = np.logical_and(step >= history_frames, step < 50)
        post_step = step[future_mask] - history_frames
        post_traj = traj[future_mask]
        feature_y[post_step] = post_traj
        has_pred[post_step] = 1

        feature_x = np.zeros((history_frames, 6), np.float32)
        origin_object_xy = np.full((history_frames, 2), np.inf)
        obs_mask = step < history_frames
        his_step = step[obs_mask]
        his_traj = traj[obs_mask]
        for i in range(len(his_step)):
            if his_step[i] == 19 - (len(his_step) - 1) + i:
                break
        his_step = his_step[i:]
        his_traj = his_traj[i:]

        origin_object_xy[his_step] = his_traj
        # origin_objects_xy_19.append(his_traj[-1:])

        his_traj = np.matmul(rot, (his_traj - orig.reshape(-1, 2)).T).T
        feature_x[his_step, :2] = his_traj
        feature_x[his_step, -1] = 1

        ctrs.append(feature_x[-1, :2].copy())
        feature_x[1:, 2:4] = feature_x[1:, :2] - feature_x[:-1, :2]
        # feature_x[1:, :2] -= feature_x[:-1, :2]
        feature_x[:, 4] = np.arctan2(feature_x[:, 0], feature_x[:, 1])
        if his_step[0] < 20:
            feature_x[his_step[0], 2:4] = 0

        origin_objects_xy.append(origin_object_xy)
        features_x.append(feature_x)
        features_y.append(feature_y)
        has_preds.append(has_pred)

    # origin_objects_xy_19 = np.concatenate(origin_objects_xy_19, 0)  # (N,2)
    origin_objects_xy = np.asarray(origin_objects_xy, np.float32)  # (N,20,2)
    features_x = np.asarray(features_x, np.float32)  # (N,20,3)
    ctrs = np.asarray(ctrs, np.float32)  # (N,2)
    features_y = np.asarray(features_y, np.float32)  # (N,30,2)
    has_preds = np.asarray(has_preds, bool)  # (N,30)

    data['features_x'] = torch.from_numpy(features_x).float()
    data['orig'] = torch.from_numpy(orig).float()
    data['ctrs'] = torch.from_numpy(ctrs).float()
    data['rot'] = torch.from_numpy(rot).float()
    data['features_y'] = torch.from_numpy(features_y).float()
    data['has_preds'] = torch.from_numpy(has_preds).bool()
    data["origin_objects_xy"] = torch.from_numpy(origin_objects_xy).float()
    return data, data_trajs, data_steps


def process_lane(data, data_trajs, data_steps):
    a = data["origin_objects_xy"].reshape(-1, 2)
    x = torch.unique(a[:, 0])
    y = torch.unique(a[:, 1])
    left_boundary, down_boundary = min(x), min(y)
    right_boundary = torch.sort(x, descending=True).values[1] if torch.sort(x, descending=True).values[
                                                                     0] == torch.inf else \
        torch.sort(x, descending=True).values[0]
    up_boundary = torch.sort(y, descending=True).values[1] if torch.sort(y, descending=True).values[0] == torch.inf else \
        torch.sort(y, descending=True).values[0]

    all_lane_ids = am.get_lane_ids_in_xy_bbox((left_boundary + right_boundary) / 2,
                                              (down_boundary + up_boundary) / 2, data["city"],
                                              max((right_boundary - left_boundary) / 2,
                                                  (up_boundary - down_boundary) / 2) + 50)
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
    lane_heading = torch.zeros(num_lanes, dtype=torch.float)
    lane_length = torch.zeros(num_lanes, dtype=torch.float)
    lane_is_intersection = torch.zeros(num_lanes, dtype=torch.uint8)
    lane_turn_direction = torch.zeros(num_lanes, dtype=torch.uint8)
    lane_traffic_control = torch.zeros(num_lanes, dtype=torch.uint8)

    num_centerlines = torch.zeros(num_lanes, dtype=torch.long)
    centerline_position = [None] * num_lanes
    centerline_heading = [None] * num_lanes
    centerline_length = [None] * num_lanes

    lane_adjacent_edge_index = []
    lane_predecessor_edge_index = []
    lane_successor_edge_index = []
    for lane_id in lane_ids:
        lane_idx = lane_ids.index(lane_id)

        centerlines = lane_dict[lane_id].centerline
        num_centerlines[lane_idx] = centerlines.size(0) - 1
        centerline_position[lane_idx] = (centerlines[1:] + centerlines[:-1]) / 2
        centerline_vectors = centerlines[1:] - centerlines[:-1]
        centerline_length[lane_idx], centerline_heading[lane_idx] = compute_angles_lengths_2D(centerline_vectors)

        lane_length[lane_idx] = centerline_length[lane_idx].sum()
        center_index = int(num_centerlines[lane_idx] / 2)
        lane_position[lane_idx] = centerlines[center_index]
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
    data['lane_length'] = lane_length
    data['lane_heading'] = lane_heading
    data['lane_is_intersection'] = lane_is_intersection
    data['lane_turn_direction'] = lane_turn_direction
    data['lane_traffic_control'] = lane_traffic_control

    data['centerline_num_nodes'] = num_centerlines.sum().item()
    data['centerline_position'] = torch.cat(centerline_position, dim=0)
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

    trajs_ = []
    steps_ = []

    for traj, step in zip(data_trajs, data_steps):
        if 19 not in step:
            continue
        trajs_.append(traj)
        steps_.append(step)
    a2a_heading = np.zeros((len(trajs_), len(trajs_)), np.float32)
    l2a_heading = np.zeros((len(trajs_), lane_position.shape[0]), np.float32)

    k = -1
    for traj, step in zip(trajs_, steps_):
        k = k + 1
        orig = traj[np.where(step == 19)[0]].copy().astype(np.float32)
        if len(np.where(step == 18)[0]) == 0:
            orig_1 = np.zeros((1, 2))
        else:
            orig_1 = traj[np.where(step == 18)[0]].copy().astype(np.float32)
        pre = orig_1 - orig
        theta = np.pi - np.arctan2(pre[0][1], pre[0][0])
        rot = np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]], np.float32)

        j = -1
        for traj, step in zip(trajs_, steps_):
            j = j + 1
            obs_mask = step < history_frames
            his_step = step[obs_mask]
            his_traj = traj[obs_mask]
            for i in range(len(his_step)):
                if his_step[i] == 19 - (len(his_step) - 1) + i:
                    break
            his_traj = his_traj[i:]
            his_traj = np.matmul(rot, (his_traj - orig.reshape(-1, 2)).T).T
            a2a_heading[k, j] = np.arctan2(his_traj[-1, 0], his_traj[-1, 1])

        m = -1
        for lane_id in all_lane_ids:
            m = m + 1
            lane = am.city_lane_centerlines_dict[data["city"]][lane_id]
            lane = copy.deepcopy(lane)
            centerline = np.matmul(rot, (lane.centerline - orig.reshape(-1, 2)).T).T
            # print(centerline.shape)
            num_centerlines = centerline.shape[0] - 1
            center_index = int(num_centerlines / 2)
            l2a_heading[k, m] = np.arctan2(centerline[center_index][0], centerline[center_index][1])

    data['a2a_heading'] = torch.from_numpy(a2a_heading).float()
    data['l2a_heading'] = torch.from_numpy(l2a_heading).float()

    return data


if __name__ == "__main__":
    for j in range(18, 22):
        avl = ArgoverseForecastingLoader(
            r"/home/ME_4012_DATA2/xc1/wuzixuan/train_data/train_data_" + str(j) + "0000_" + str(j + 1) + "0000")
        for i in tqdm(range(10000)):
            process_data(i, avl)
    # data = torch.load("/home/ME_4012_DATA2/xc1/wuzixuan/pre_train_data_my/101481.pt")
    # print(data["lane_num_nodes"])
    # print(data["features_x"].shape)
    # print(data["ctrs"].shape)
    # print(data["lane_position"].shape)
    # object_lane_mask = torch.cdist(data["ctrs"], data["lane_position"], p=2)  # (B,Nax,Lax)
    # print(object_lane_mask)
