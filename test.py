import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import Model
from utils import gpu
import os
from argoverse.evaluation.competition_util import generate_forecasting_h5

batch_size = 16
hidden_size = 64
heads = 8
lr = 0.0004
epochs = 100
dropout = 0.2
num_historical_steps = 20
future_frames = 30
model_save_path = "/home/ME_4012_DATA2/xc1/wuzixuan/argoverse_3/model_save_path/"
# test_log = "/home/ME_4012_DATA2/xc1/wuzixuan/argoverse_4/test_log.txt"
device = torch.device("cuda:3")


def collate_fn(batch):
    # batch = from_numpy(batch)
    return_batch = dict()
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch


class ArgoverseDatasets(Dataset):
    def __init__(self, folder_path):
        super(ArgoverseDatasets, self).__init__()
        self.folder_path = folder_path
        self.file_list = os.listdir(self.folder_path)
        if ".ipynb_checkpoints" in self.file_list:
            self.file_list.remove(".ipynb_checkpoints")

    def __getitem__(self, idx):
        return torch.load(self.folder_path + self.file_list[idx])

    def __len__(self):
        return len(self.file_list)


def transfer_device(data):
    for keys in list(data.keys()):
        data[keys] = gpu(data[keys], device)
    return data


if __name__ == "__main__":
    test_datasets = ArgoverseDatasets("/home/ME_4012_DATA2/xc1/wuzixuan/new_pre_test_data/")
    test_loader = DataLoader(test_datasets, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                             drop_last=False)

    model = Model(hidden_size, heads, 3, num_historical_steps, future_frames, device)
    model.load_state_dict(torch.load("/home/ME_4012_DATA2/xc1/wuzixuan/argoverse_3/model_save_path/epoch_55.pth"))
    model.to(device)

    preds = {}
    gts = {}
    cities = {}
    for i, data in enumerate(test_loader):
        data = transfer_device(data)
        with torch.no_grad():
            out = model(data)
        results = [x[0:1].detach().cpu().numpy() for x in out["reg"]]  # [(6,30,2),(6,30,2),(6,30,2)...]
        # print(len(results))
        # print(results[0].shape)
        for i, (argo_idx, pred_traj) in enumerate(zip(data["argo_id"], results)):
            # print(pred_traj.squeeze())
            preds[argo_idx] = pred_traj.squeeze()
            cities[argo_idx] = data["city"][i]
            gts[argo_idx] = data["features_y"][i][0] if "features_y" in data else None

    generate_forecasting_h5(preds, "/home/ME_4012_DATA2/xc1/wuzixuan/argoverse_3/submit1.h5")  # this might take a while
    print("end................")

