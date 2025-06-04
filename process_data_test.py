import lanelet2
import os
import numpy as np
import pandas as pd
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import math
from lanelet2.projection import UtmProjector
from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData

projector = UtmProjector(lanelet2.io.Origin(0, 0))
traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                              lanelet2.traffic_rules.Participants.Vehicle)


def compute_angles_lengths_2D(vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    length = torch.norm(vectors, dim=-1)
    theta = torch.atan2(vectors[..., 1], vectors[..., 0])
    return length, theta


def wrap_angle(angle: torch.Tensor, min_val: float = -math.pi, max_val: float = math.pi) -> torch.Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)


def get_index_of_A_in_B(list_A: Optional[List[Any]], list_B: Optional[List[Any]]) -> List[int]:
    if not list_A or not list_B:
        return []

    set_B = set(list_B)
    indices = [list_B.index(i) for i in list_A if i in set_B]

    return indices


def process(raw_paths):
    for raw_path in tqdm(raw_paths):
        # map
        raw_dir, raw_file_name = os.path.split(raw_path)
        # print("raw_dir:",
        #       raw_dir)  # /home/ME_4012_DATA2/xc1/wuzixuan/interaction/INTERACTION-Dataset-DR-multi-v1_2/train
        # print("raw_file_name:", raw_file_name)  # DR_USA_Intersection_MA_train.csv
        scenario_name = os.path.splitext(raw_file_name)[0]
        scenario_name = '_'.join(scenario_name.split('_')[:-1])
        # print("scenario_name:", scenario_name)  # DR_USA_Intersection_MA
        base_dir = os.path.dirname(raw_dir)
        map_path = os.path.join(base_dir, 'maps', scenario_name + '.osm')
        map_api = lanelet2.io.load(map_path, projector)
        routing_graph = lanelet2.routing.RoutingGraph(map_api, traffic_rules)

        # agent
        df = pd.read_csv(raw_path)
        for case_id, case_df in tqdm(df.groupby('case_id')):
            data = dict()
            data['scenario_name'] = scenario_name
            data['case_id'] = int(case_id)
            data = get_features(case_df, map_api, routing_graph, data)
            torch.save(data, os.path.join(processed_dir, scenario_name + '_' + str(int(case_id)) + '.pt'))


def get_features(df: pd.DataFrame,
                 map_api,
                 routing_graph, data):
    # data = {
    #     'agent': {},
    #     'lane': {},
    #     'polyline': {},
    #     ('polyline', 'lane'): {},
    #     ('lane', 'lane'): {}
    # }

    # MAP
    lane_ids = []
    for lane in map_api.laneletLayer:
        lane_ids.append(lane.id)

    num_lanes = len(lane_ids)
    lane_id = torch.zeros(num_lanes, dtype=torch.float)
    lane_position = torch.zeros(num_lanes, 2, dtype=torch.float)
    lane_heading = torch.zeros(num_lanes, dtype=torch.float)
    lane_length = torch.zeros(num_lanes, dtype=torch.float)

    num_polylines = torch.zeros(num_lanes, dtype=torch.long)
    polyline_position: List[Optional[torch.Tensor]] = [None] * num_lanes
    polyline_heading: List[Optional[torch.Tensor]] = [None] * num_lanes
    polyline_length: List[Optional[torch.Tensor]] = [None] * num_lanes
    polyline_side: List[Optional[torch.Tensor]] = [None] * num_lanes

    lane_left_neighbor_edge_index = []
    lane_right_neighbor_edge_index = []
    lane_predecessor_edge_index = []
    lane_successor_edge_index = []

    for lane in map_api.laneletLayer:
        lane_idx = lane_ids.index(lane.id)
        lane_id[lane_idx] = lane.id

        points = [np.array([pt.x, pt.y]) for pt in lane.centerline]
        centerline = torch.from_numpy(np.array(points)).float()

        center_index = int((centerline.size(0) - 1) / 2)
        lane_position[lane_idx] = centerline[center_index, :2]
        lane_heading[lane_idx] = torch.atan2(centerline[center_index + 1, 1] - centerline[center_index, 1],
                                             centerline[center_index + 1, 0] - centerline[center_index, 0])
        lane_length[lane_idx] = torch.norm(centerline[1:] - centerline[:-1], p=2, dim=-1).sum()

        points = [np.array([pt.x, pt.y]) for pt in lane.leftBound]
        left_boundary = torch.from_numpy(np.array(points)).float()
        points = [np.array([pt.x, pt.y]) for pt in lane.rightBound]
        right_boundary = torch.from_numpy(np.array(points)).float()
        left_vector = left_boundary[1:] - left_boundary[:-1]
        right_vector = right_boundary[1:] - right_boundary[:-1]
        centerline_vector = centerline[1:] - centerline[:-1]
        polyline_position[lane_idx] = torch.cat \
            ([(left_boundary[1:] + left_boundary[:-1]) / 2, (right_boundary[1:] + right_boundary[:-1]) / 2,
              (centerline[1:] + centerline[:-1]) / 2], dim=0)
        polyline_length[lane_idx], polyline_heading[lane_idx] = compute_angles_lengths_2D \
            (torch.cat([left_vector, right_vector, centerline_vector], dim=0))
        num_left_polyline = len(left_vector)
        num_right_polyline = len(right_vector)
        num_centerline_polyline = len(centerline_vector)
        polyline_side[lane_idx] = torch.cat(
            [torch.full((num_left_polyline,), _polyline_side.index('left'), dtype=torch.uint8),
             torch.full((num_right_polyline,), _polyline_side.index('right'), dtype=torch.uint8),
             torch.full((num_centerline_polyline,), _polyline_side.index('center'), dtype=torch.uint8)], dim=0)

        num_polylines[lane_idx] = num_left_polyline + num_right_polyline + num_centerline_polyline

        lane_left_neighbor_lane = routing_graph.left(lane)
        lane_left_neighbor_id = [lane_left_neighbor_lane.id] if lane_left_neighbor_lane else []
        lane_left_neighbor_idx = get_index_of_A_in_B(lane_left_neighbor_id, lane_ids)
        if len(lane_left_neighbor_idx) != 0:
            edge_index = torch.stack([torch.tensor(lane_left_neighbor_idx, dtype=torch.long),
                                      torch.full((len(lane_left_neighbor_idx),), lane_idx, dtype=torch.long)],
                                     dim=0)
            lane_left_neighbor_edge_index.append(edge_index)
        lane_right_neighbor_lane = routing_graph.right(lane)
        lane_right_neighbor_id = [lane_right_neighbor_lane.id] if lane_right_neighbor_lane else []
        lane_right_neighbor_idx = get_index_of_A_in_B(lane_right_neighbor_id, lane_ids)
        if len(lane_right_neighbor_idx) != 0:
            edge_index = torch.stack([torch.tensor(lane_right_neighbor_idx, dtype=torch.long),
                                      torch.full((len(lane_right_neighbor_idx),), lane_idx, dtype=torch.long)],
                                     dim=0)
            lane_right_neighbor_edge_index.append(edge_index)
        lane_predecessor_lanes = routing_graph.previous(lane)
        lane_predecessor_ids = [ll.id for ll in lane_predecessor_lanes] if lane_predecessor_lanes else []
        lane_predecessor_idx = get_index_of_A_in_B(lane_predecessor_ids, lane_ids)
        if len(lane_predecessor_idx) != 0:
            edge_index = torch.stack([torch.tensor(lane_predecessor_idx, dtype=torch.long),
                                      torch.full((len(lane_predecessor_idx),), lane_idx, dtype=torch.long)], dim=0)
            lane_predecessor_edge_index.append(edge_index)
        lane_successor_lanes = routing_graph.following(lane)
        lane_successor_ids = [ll.id for ll in lane_successor_lanes] if lane_successor_lanes else []
        lane_successor_idx = get_index_of_A_in_B(lane_successor_ids, lane_ids)
        if len(lane_successor_idx) != 0:
            edge_index = torch.stack([torch.tensor(lane_successor_idx, dtype=torch.long),
                                      torch.full((len(lane_successor_idx),), lane_idx, dtype=torch.long)], dim=0)
            lane_successor_edge_index.append(edge_index)

    data['lane_id'] = lane_id
    data['lane_num_nodes'] = num_lanes
    data['lane_true_position'] = lane_position
    data['lane_position'] = lane_position
    data['lane_length'] = lane_length
    data['lane_heading'] = lane_heading

    data['polyline_num_nodes'] = num_polylines.sum().item()
    data['polyline_position'] = torch.cat(polyline_position, dim=0)
    data['polyline_true_position'] = torch.cat(polyline_position, dim=0)
    data['polyline_heading'] = torch.cat(polyline_heading, dim=0)
    data['polyline_length'] = torch.cat(polyline_length, dim=0)
    data['polyline_side'] = torch.cat(polyline_side, dim=0)

    polyline_to_lane_edge_index = torch.stack([torch.arange(num_polylines.sum(), dtype=torch.long),
                                               torch.arange(num_lanes, dtype=torch.long).repeat_interleave
                                               (num_polylines)], dim=0)
    data['polyline_to_lane_edge_index'] = polyline_to_lane_edge_index

    if len(lane_left_neighbor_edge_index) != 0:
        lane_left_neighbor_edge_index = torch.cat(lane_left_neighbor_edge_index, dim=1)
    else:
        lane_left_neighbor_edge_index = torch.tensor([[], []], dtype=torch.long)
    if len(lane_right_neighbor_edge_index) != 0:
        lane_right_neighbor_edge_index = torch.cat(lane_right_neighbor_edge_index, dim=1)
    else:
        lane_right_neighbor_edge_index = torch.tensor([[], []], dtype=torch.long)
    if len(lane_predecessor_edge_index) != 0:
        lane_predecessor_edge_index = torch.cat(lane_predecessor_edge_index, dim=1)
    else:
        lane_right_neighbor_edge_index = torch.tensor([[], []], dtype=torch.long)
    if len(lane_successor_edge_index) != 0:
        lane_successor_edge_index = torch.cat(lane_successor_edge_index, dim=1)
    else:
        lane_successor_edge_index = torch.tensor([[], []], dtype=torch.long)

    data['left_neighbor_edge_index'] = lane_left_neighbor_edge_index
    data['right_neighbor_edge_index'] = lane_right_neighbor_edge_index
    data['predecessor_edge_index'] = lane_predecessor_edge_index
    data['successor_edge_index'] = lane_successor_edge_index

    max_lane_x, min_lane_x = max(data['lane_position'][:, 0]), min(data['lane_position'][:, 0])
    max_lane_y, min_lane_y = max(data['lane_position'][:, 1]), min(data['lane_position'][:, 1])

    max_center_x, min_center_x = max(data['polyline_position'][:, 0]), min(data['polyline_position'][:, 0])
    max_center_y, min_center_y = max(data['polyline_position'][:, 1]), min(data['polyline_position'][:, 1])

    max_map_x = max_lane_x if max_lane_x > max_center_x else max_center_x
    min_map_x = min_center_x if min_lane_x > min_center_x else min_lane_x
    max_map_y = max_lane_y if max_lane_y > max_center_y else max_center_y
    min_map_y = min_center_y if min_lane_y > min_center_y else min_lane_y
    max_x, min_x, max_y, min_y = max(df["x"]), min(df["x"]), max(df["y"]), min(df["y"])

    max_all_x = max_x if max_x > max_map_x else max_map_x
    # max_all_x = torch.tensor(max_all_x)
    min_all_x = min_map_x if min_x > min_map_x else min_x
    # min_all_x = torch.tensor(min_all_x)
    max_all_y = max_y if max_y > max_map_y else max_map_y
    # max_all_y = torch.tensor(max_all_y)
    min_all_y = min_map_y if min_y > min_map_y else min_y
    # min_all_y = torch.tensor(min_all_y)
    data["max_all_x"] = torch.tensor(max_all_x) if type(max_all_x)==float else max_all_x
    data["min_all_x"] = torch.tensor(min_all_x) if type(min_all_x)==float else min_all_x
    data["max_all_y"] = torch.tensor(max_all_y) if type(max_all_y)==float else max_all_y
    data["min_all_y"] = torch.tensor(min_all_y) if type(min_all_y)==float else min_all_y
    # print(max_all_x,min_all_x, max_all_y, min_all_y)

    data['lane_position'] = (data['lane_position'] - torch.tensor([min_all_x, min_all_y])) / (
        torch.tensor([max_all_x - min_all_x, max_all_y - min_all_y]))

    data['polyline_position'] = (data['polyline_position'] - torch.tensor([min_all_x, min_all_y])) / (
        torch.tensor([max_all_x - min_all_x, max_all_y - min_all_y]))

    # AGENT
    # filter out actors that are unseen during the historical time steps
    timestep_ids = list(np.sort(df['timestamp_ms'].unique()))
    historical_timestamps = timestep_ids[:num_historical_steps]
    historical_df = df[df['timestamp_ms'].isin(historical_timestamps)]
    agent_ids = list(historical_df['track_id'].unique())
    num_agents = len(agent_ids)
    df = df[df['track_id'].isin(agent_ids)]

    # initialization
    agent_id = torch.zeros(num_agents, dtype=torch.uint8)
    visible_mask = torch.zeros(num_agents, num_steps, dtype=torch.bool)
    agent_position = torch.zeros(num_agents, num_steps, 2, dtype=torch.float)
    agent_true_position = torch.zeros(num_agents, num_steps, 2, dtype=torch.float)
    agent_heading = torch.zeros(num_agents, num_steps, 1, dtype=torch.float)
    agent_velocity = torch.zeros(num_agents, num_steps, 2, dtype=torch.float)
    agent_velocity_length = torch.zeros(num_agents, num_historical_steps, 1, dtype=torch.float)
    agent_velocity_theta = torch.zeros(num_agents, num_historical_steps, 1, dtype=torch.float)
    agent_length = torch.zeros(num_agents, num_historical_steps, 1, dtype=torch.float)
    agent_width = torch.zeros(num_agents, num_historical_steps, 1, dtype=torch.float)
    agent_type = torch.zeros(num_agents, dtype=torch.uint8)
    agent_category = torch.zeros(num_agents, dtype=torch.uint8)
    agent_interset = torch.zeros(num_agents, dtype=torch.uint8)

    for track_id, track_df in df.groupby('track_id'):
        agent_idx = agent_ids.index(track_id)
        agent_id[agent_idx] = track_id
        agent_steps = [timestep_ids.index(timestamp) for timestamp in track_df['timestamp_ms']]

        visible_mask[agent_idx, agent_steps] = True

        agent_type_name = track_df['agent_type'].values[0]
        agent_type[agent_idx] = torch.tensor(_agent_type.index(agent_type_name), dtype=torch.uint8)

        if agent_type_name == 'car':
            agent_length[agent_idx, :, 0] = track_df['length'].values[0]
            agent_width[agent_idx, :, 0] = track_df['width'].values[0]

        if 'track_to_predict' in track_df.columns:
            agent_category[agent_idx] = torch.tensor(track_df['track_to_predict'].values[0], dtype=torch.uint8)
            agent_interset[agent_idx] = torch.tensor(track_df['interesting_agent'].values[0], dtype=torch.uint8)
        elif agent_type_name == 'car':
            agent_category[agent_idx] = torch.tensor(1, dtype=torch.uint8)

        agent_position[agent_idx, agent_steps] = torch.from_numpy \
            (np.stack([(track_df['x'].values - data["min_all_x"].numpy()) / (data["max_all_x"].numpy() - data["min_all_x"].numpy()),
                       (track_df['y'].values - data["min_all_y"].numpy()) / (data["max_all_y"].numpy() - data["min_all_y"].numpy())],
                      axis=-1)).float()
        agent_true_position[agent_idx, agent_steps] = torch.from_numpy(
            np.stack([track_df['x'].values, track_df['y'].values], axis=-1)).float()

        agent_velocity[agent_idx, agent_steps] = torch.from_numpy \
            (np.stack([track_df['vx'].values, track_df['vy'].values], axis=-1)).float()
        velocity_length, velocity_theta = compute_angles_lengths_2D(agent_velocity[agent_idx])
        agent_velocity_length[agent_idx, :, 0] = velocity_length[:num_historical_steps]

        if agent_type_name == 'car':
            agent_heading[agent_idx, agent_steps, 0] = torch.from_numpy(track_df['psi_rad'].values).float()
            agent_velocity_theta[agent_idx, :, 0] = wrap_angle \
                (velocity_theta[:num_historical_steps] - agent_heading[agent_idx, :num_historical_steps, 0])
        else:
            agent_heading[agent_idx, :, 0] = velocity_theta
            agent_velocity_theta[agent_idx, :, 0] = 0

    data['agent_id'] = agent_id
    data['agent_num_nodes'] = num_agents
    data['agent_visible_mask'] = visible_mask
    data['agent_position'] = agent_position[:, :num_historical_steps]
    data['agent_ctrs'] = agent_position[:, num_historical_steps - 1]
    data['agent_true_position'] = agent_true_position
    data['agent_heading'] = agent_heading
    data['agent_velocity'] = agent_velocity
    data['agent_velocity_length'] = agent_velocity_length
    data['agent_velocity_theta'] = agent_velocity_theta
    data['agent_length'] = agent_length
    data['agent_width'] = agent_width
    data['agent_type'] = agent_type
    data['agent_category'] = agent_category
    data['agent_interest'] = agent_interset

    return data


num_historical_steps = 10
num_future_steps = 0
num_steps = num_historical_steps + num_future_steps

projector = UtmProjector(lanelet2.io.Origin(0, 0))
traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                              lanelet2.traffic_rules.Participants.Vehicle)

_agent_type = ['car', 'pedestrian/bicycle']
_polyline_side = ['left', 'center', 'right']

root = "/home/ME_4012_DATA2/xc1/wuzixuan/InteractionDataset/INTERACTION-Dataset-DR-multi-v1_2"
_directory = 'test_multi-agent'
processed_dir = os.path.join(root, _directory + '_processed')

listdir = os.listdir(os.path.join(root, "test_multi-agent"))
raw_paths = []
for i in listdir:
    raw_paths.append(os.path.join(root, "test_multi-agent", i))

data = process(raw_paths)
