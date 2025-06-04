import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import Model
from Loss import Loss, PostProcess
from utils import gpu
import time
from argoverse.evaluation.eval_forecasting import compute_forecasting_metrics
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

batch_size = 8
hidden_size = 64
heads = 8
lr = 0.0004
epochs = 100
dropout = 0.2
num_historical_steps = 20
future_frames = 30
model_save_path = "/home/ME_4012_DATA2/xc1/wuzixuan/argoverse_3/model_save_path/"
train_log = "/home/ME_4012_DATA2/xc1/wuzixuan/argoverse_3/nohup.txt"
device = torch.device("cuda:1")


def collate_fn(batch):
    # batch = from_numpy(batch)
    return_batch = dict()
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch


def transfer_device(data):
    for keys in list(data.keys()):
        data[keys] = gpu(data[keys], device)
    return data


class ArgoverseDatasets(Dataset):
    def __init__(self, folder_path):
        super(ArgoverseDatasets, self).__init__()
        self.folder_path = folder_path
        self.file_list = os.listdir(self.folder_path)[:10000]
        if ".ipynb_checkpoints" in self.file_list:
            self.file_list.remove(".ipynb_checkpoints")

    def __getitem__(self, idx):
        return torch.load(self.folder_path + self.file_list[idx])

    def __len__(self):
        return len(self.file_list)


def display(out, gt_preds, has_preds, loss_item):
    preds = [x[0:1].detach().cpu().numpy() for x in out["reg"]]
    cls = [x[0:1].detach().cpu().numpy() for x in out["cls"]]
    gt_preds = [x[0:1].cpu().numpy() for x in gt_preds]
    has_preds = [x[0:1].cpu().numpy() for x in has_preds]

    preds = np.concatenate(preds, 0)  # (B,6,30,2)
    cls = np.concatenate(cls, 0)  # (B,6)
    gt_preds = np.concatenate(gt_preds, 0)  # (B,30,2)
    has_preds = np.concatenate(has_preds, 0)  # (B,30)

    batch = preds.shape[0]

    dist = []
    for j in range(6):
        dist.append(
            np.sum(np.sqrt(np.sum((preds[:, j] - gt_preds) ** 2, axis=2)) * has_preds, axis=1) / (has_preds.sum(1)))

    dist = np.array(dist).T  # (B,6)

    min_dist, min_idcs = dist.min(1), dist.argmin(1)
    ade = np.sum(min_dist) / preds.shape[0]
    final_index = has_preds.sum(1) - 1
    fde = np.sum(np.sqrt(
        ((preds[np.arange(batch), min_idcs, final_index] - gt_preds[np.arange(batch), final_index]) ** 2).sum(1))) / \
          preds.shape[0]

    log = "ade:" + str(ade) + ",fde:" + str(fde) + ", loss:" + str(loss_item)
    with open(train_log, "a") as file:
        file.write(log)
        file.write("\n")



if __name__ == "__main__":

    train_datasets = ArgoverseDatasets("/home/ME_4012_DATA2/xc1/wuzixuan/new_pre_train_data/")
    val_datasets = ArgoverseDatasets("/home/ME_4012_DATA2/xc1/wuzixuan/new_pre_val_data/")
    train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                              drop_last=True)
    val_loader = DataLoader(val_datasets, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

    model = Model(hidden_size, heads, 3, num_historical_steps, future_frames, device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                           threshold=0.05, threshold_mode="abs", verbose=False)

    loss1 = Loss()
    loss2 = Loss()
    post_process = PostProcess()

    val_ade = []
    val_fde = []
    val_mr = []
    for epoch in range(1, epochs + 1):
        with open(train_log, "a") as file:
            file.write("Epoch:" + str(epoch))
            file.write("\n")
            file.write("Learning Rate:" + str(optimizer.param_groups[0]["lr"]))
            file.write("\n")

        model.train()
        # metrics = dict()
        epoch_start_time = time.perf_counter()
        for i, data in enumerate(train_loader):
            data = transfer_device(data)
            model_out = model(data)
            loss_out = loss1(model_out, data["gt_preds"], data["has_preds"])
            optimizer.zero_grad()
            loss_out["loss"].backward()
            optimizer.step()

            if i % 1000 == 0:
                # post_process.display(metrics)
                display(model_out, data["gt_preds"], data["has_preds"], loss_out["loss"].item())

        # val
        torch.save(model.state_dict(), model_save_path + "epoch_" + str(epoch) + ".pth")
        model.eval()
        preds = {}
        gts = {}
        cities = {}
        val_loss = []
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                data = transfer_device(data)
                out = model(data)
                val_loss_out = loss2(out, data["gt_preds"], data["has_preds"])
                val_loss.append(val_loss_out["loss"].item())
                results = [x[0:1].detach().cpu().numpy() for x in out["reg"]]  # [(6,30,2),(6,30,2),(6,30,2)...]
                for i, (argo_idx, pred_traj) in enumerate(zip(data["argo_id"], results)):
                    preds[argo_idx] = pred_traj.squeeze()
                    cities[argo_idx] = data["city"][i]
                    gts[argo_idx] = data["gt_preds"][i][0] if "gt_preds" in data else None
        metric_results = compute_forecasting_metrics(preds, gts, cities, 6, 30, 2)
        mean_loss = sum(val_loss) / len(val_loss)
        log = "min_ade:" + str(metric_results["minADE"]) + ",min_fde:" + str(metric_results["minFDE"]) + ",mr:" + str(
            metric_results["MR"]) + ", val_loss:" + str(mean_loss)
        with open(train_log, "a") as file:
            file.write(log)
            file.write("\n")
        val_ade.append(metric_results["minADE"])
        val_fde.append(metric_results["minFDE"])
        val_mr.append(metric_results["MR"])
        epoch_end_time = time.perf_counter()
        spend_time = epoch_end_time - epoch_start_time
        with open(train_log, "a") as file:
            file.write("epoch spend time:" + str(spend_time))
            file.write("\n")
            file.write("best epoch:" + str(val_ade.index(min(val_ade))))
            file.write("\n")
            file.write("***************************************************")
            file.write("\n")
        scheduler.step(mean_loss)
