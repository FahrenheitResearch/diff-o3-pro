import yaml, torch, time, os
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import MSELoss
from tqdm import tqdm
from models.unet_attention_fixed import UNetAttnSmall
from hrrr_dataset.hrrr_data import HRRRDataset
import utils.metrics as metrics

def load_cfg(path="configs/default.yaml"):
    return yaml.safe_load(open(path))

def main():
    cfg = load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_train = HRRRDataset(
        Path(cfg["data"]["zarr"]),
        cfg["data"]["variables"],
        cfg["training"]["lead_hours"],
        Path(cfg["data"]["stats"])
    )
    dl = DataLoader(ds_train, cfg["training"]["batch_size"], shuffle=True,
                    num_workers=cfg["training"]["num_workers"], pin_memory=True)

    model = UNetAttnSmall(len(cfg["data"]["variables"]), len(cfg["data"]["variables"])).to(device)
    opt = AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=1e-4)
    criterion = MSELoss()

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        pbar = tqdm(dl, desc=f"epoch {epoch}")
        running = 0
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            yhat = model(x)
            loss = criterion(yhat, y)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
            pbar.set_postfix(loss=running/ (pbar.n+1))
        torch.save(model.state_dict(), f"ckpt_epoch{epoch:03}.pt")

if __name__ == "__main__":
    main()