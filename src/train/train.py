import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from src.train.dataset import load_patch_pairs, SRGANDataset
from src.train.model import Generator, Discriminator


# =========================
# KONFIGURASI
# =========================
START_EPOCH = 6      # Ubah sesuai anggota
END_EPOCH   = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# =========================
# LOAD DATASET
# =========================
splits = load_patch_pairs()

train_lr, train_hr = splits["train"]
val_lr, val_hr     = splits["val"]

train_dataset = SRGANDataset(train_lr, train_hr)
val_dataset   = SRGANDataset(val_lr, val_hr)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

print("Train batches:", len(train_loader))
print("Val batches  :", len(val_loader))


# =========================
# MODEL
# =========================
G = Generator(upscale_factor=2).to(device)
D = Discriminator().to(device)


# =========================
# LOSS & OPTIMIZER
# =========================
criterion_content = nn.L1Loss().to(device)
criterion_adv = nn.BCEWithLogitsLoss().to(device)

optimizer_G = Adam(G.parameters(), lr=1e-4, betas=(0.9, 0.999))
optimizer_D = Adam(D.parameters(), lr=1e-4, betas=(0.9, 0.999))


# =========================
# RESUME SAFE MODE
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
checkpoint_dir = os.path.join(BASE_DIR, "outputs", "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

def is_discriminator_state_valid(state_dict):
    """Return False jika file ini bukan Discriminator SRGAN (contoh: VGG/classifier)."""
    invalid = any(
        k.startswith("features") or k.startswith("classifier")
        for k in state_dict.keys()
    )
    return not invalid


if START_EPOCH > 1:
    last = START_EPOCH - 1
    print(f"\n========= SAFE RESUME FROM EPOCH {last} =========")

    g_path = os.path.join(checkpoint_dir, f"G_epoch_{last}.pth")
    d_path = os.path.join(checkpoint_dir, f"D_epoch_{last}.pth")

    # ============ LOAD G (selalu coba) ============
    if os.path.exists(g_path):
        try:
            G.load_state_dict(torch.load(g_path, map_location=device))
            print("G loaded successfully.")
        except Exception as e:
            print("FAILED loading G:", e)
            raise RuntimeError("Cannot resume without a valid Generator checkpoint.")

    else:
        print("WARNING: No G checkpoint found. Cannot resume training.")
        raise FileNotFoundError("G_epoch file missing.")

    # ============ LOAD D (AMANKAN) ============
    if os.path.exists(d_path):
        try:
            stateD = torch.load(d_path, map_location=device)
            if is_discriminator_state_valid(stateD):
                D.load_state_dict(stateD)
                print("D loaded successfully.")
            else:
                print("WARNING: D checkpoint INVALID (looks like VGG or other model). RESET D.")
        except Exception as e:
            print("WARNING: Failed loading D, RESET D:", e)
    else:
        print("No D checkpoint found — RESET D.")

    print("SAFE RESUME COMPLETE.\n")


# =========================
# TRAINING LOOP
# =========================
lambda_adv = 1e-3

for epoch in range(START_EPOCH, END_EPOCH + 1):

    G.train()
    D.train()

    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{END_EPOCH}")

    for lr, hr in loop:

        lr = lr.to(device)
        hr = hr.to(device)

        bs = lr.size(0)
        real_labels = torch.ones(bs, device=device)
        fake_labels = torch.zeros(bs, device=device)

        # =========================
        # TRAIN D
        # =========================
        with torch.no_grad():
            sr = G(lr)

        out_real = D(hr).view(bs, -1).mean(dim=1)
        out_fake = D(sr.detach()).view(bs, -1).mean(dim=1)

        loss_D = criterion_adv(out_real, real_labels) + criterion_adv(out_fake, fake_labels)

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # =========================
        # TRAIN G
        # =========================
        sr = G(lr)
        out_sr = D(sr).view(bs, -1).mean(dim=1)

        loss_G_content = criterion_content(sr, hr)
        loss_G_adv = criterion_adv(out_sr, real_labels)

        loss_G = loss_G_content + lambda_adv * loss_G_adv

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        loop.set_postfix({
            "G": float(loss_G.item()),
            "D": float(loss_D.item())
        })

    # =========================
    # SAVE CHECKPOINT
    # =========================
    torch.save(G.state_dict(), os.path.join(checkpoint_dir, f"G_epoch_{epoch}.pth"))
    torch.save(D.state_dict(), os.path.join(checkpoint_dir, f"D_epoch_{epoch}.pth"))

    print(f"Checkpoint epoch {epoch} saved.\n")
