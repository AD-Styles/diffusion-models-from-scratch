"""
Generative AI with Diffusion Models — From Scratch
===================================================
NVIDIA DLI 'Generative AI with Diffusion Models' 커리큘럼을 참고해
FashionMNIST 위에서 직접 구현한 학습용 코드입니다.

핵심 개념:
  - U-Net 기본 구조 (Encoder/Decoder/Skip Connection)
  - DDPM Forward/Reverse Diffusion + Time Embedding
  - GELU, GroupNorm, RearrangePool, Sinusoidal Embedding
  - Classifier-Free Guidance (CFG)
  - 통합 Conditional Generation

데이터셋 : FashionMNIST (28×28, 10-class)
시각화 결과 (results/ 폴더, 5개):
  01_forward_diffusion.png       — 노이즈가 점점 더해지는 과정
  02_training_curve.png          — Train/Val Loss 학습 곡선
  03_reverse_trajectory.png      — 노이즈에서 이미지로 복원되는 과정
  04_generated_samples.png       — 클래스별 생성 이미지 그리드
  05_cfg_weight_comparison.png   — CFG guidance weight w 비교
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from einops.layers.torch import Rearrange
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────
# 0. 설정값 (Config)
# ──────────────────────────────────────────────────────────
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS   = 10
BATCH    = 128
LR       = 1e-3            # Adam learning rate
IMG_SZ   = 28
IMG_CH   = 1
N_CLS    = 10
T        = 400             # 총 노이즈 스텝 수
DROP_P   = 0.10            # CFG: 학습 시 클래스 정보를 떨어뜨릴 확률
T_EMBED  = 8               # Sinusoidal time embedding 차원
DOWN_CHS = (64, 64, 128)   # U-Net 다운샘플 채널 구성
SAVE_DIR = 'results'
os.makedirs(SAVE_DIR, exist_ok=True)

LABELS = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
          'Sandal',  'Shirt',   'Sneaker',  'Bag',   'Ankle Boot']

torch.manual_seed(42)
np.random.seed(42)
print(f'Device: {DEVICE}  |  T={T}  |  Epochs={EPOCHS}  |  Batch={BATCH}')

# ──────────────────────────────────────────────────────────
# 1. 데이터 준비
# ──────────────────────────────────────────────────────────
tf = transforms.Compose([
    transforms.Resize((IMG_SZ, IMG_SZ)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda x: (x * 2) - 1),   # [0,1] → [-1,1]
])
tf_val = transforms.Compose([
    transforms.Resize((IMG_SZ, IMG_SZ)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x * 2) - 1),
])
train_ds = datasets.FashionMNIST('./data', train=True,  download=True, transform=tf)
val_ds   = datasets.FashionMNIST('./data', train=False, download=True, transform=tf_val)
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  drop_last=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, drop_last=True)
print(f'Train: {len(train_ds):,} | Val: {len(val_ds):,}')

# ──────────────────────────────────────────────────────────
# 2. 노이즈 스케줄
#    β_t를 1e-4에서 0.02까지 선형으로 증가시키며 노이즈를 누적.
# ──────────────────────────────────────────────────────────
betas      = torch.linspace(1e-4, 0.02, T).to(DEVICE)
alphas     = 1.0 - betas
alpha_bar  = torch.cumprod(alphas, dim=0)

# 자주 쓰는 계수 미리 계산
sqrt_a_bar       = torch.sqrt(alpha_bar)
sqrt_one_minus   = torch.sqrt(1 - alpha_bar)
sqrt_a_inv       = torch.sqrt(1 / alphas)
pred_noise_coeff = (1 - alphas) / sqrt_one_minus


def q_sample(x_0, t, noise=None):
    """
    Forward Diffusion:
        x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε
    한 번의 계산으로 임의의 t 시점 노이즈 이미지를 얻을 수 있다.
    """
    if noise is None:
        noise = torch.randn_like(x_0)
    t = t.long()
    a = sqrt_a_bar[t][:, None, None, None]
    b = sqrt_one_minus[t][:, None, None, None]
    return a * x_0 + b * noise, noise


@torch.no_grad()
def reverse_q(x_t, t, eps):
    """
    Reverse Diffusion 한 스텝:
        μ = (1/√α_t)(x_t - (1-α_t)/√(1-ᾱ_t) · ε̂)
    t==0이면 그대로 반환, 아니면 작은 노이즈를 한 번 더 더해준다.
    """
    i = int(t[0].item())
    mu = sqrt_a_inv[i] * (x_t - pred_noise_coeff[i] * eps)
    if i == 0:
        return mu
    return mu + torch.sqrt(betas[i - 1]) * torch.randn_like(x_t)


# ──────────────────────────────────────────────────────────
# 3. U-Net 구성 블록
# ──────────────────────────────────────────────────────────
class GELUConvBlock(nn.Module):
    """Conv → GroupNorm → GELU. 가장 기본이 되는 블록."""
    def __init__(self, in_ch, out_ch, group_size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(group_size, out_ch),
            nn.GELU(),
        )
    def forward(self, x):
        return self.block(x)


class RearrangePoolBlock(nn.Module):
    """
    MaxPool 대신 사용. 공간 축을 채널 축으로 옮긴 뒤 conv로 압축.
    신경망이 직접 풀링 가중치를 학습하게 만드는 트릭.
    """
    def __init__(self, in_chs, group_size):
        super().__init__()
        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2)
        self.conv = GELUConvBlock(4 * in_chs, in_chs, group_size)
    def forward(self, x):
        return self.conv(self.rearrange(x))


class ResidualConvBlock(nn.Module):
    """잔차 연결로 체커보드 잡음을 줄여준다."""
    def __init__(self, in_chs, out_chs, group_size):
        super().__init__()
        self.conv1 = GELUConvBlock(in_chs, out_chs, group_size)
        self.conv2 = GELUConvBlock(out_chs, out_chs, group_size)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return x1 + x2


class DownBlock(nn.Module):
    """다운샘플링: 두 번 conv → RearrangePool로 크기 1/2."""
    def __init__(self, in_chs, out_chs, group_size):
        super().__init__()
        self.block = nn.Sequential(
            GELUConvBlock(in_chs, out_chs, group_size),
            GELUConvBlock(out_chs, out_chs, group_size),
            RearrangePoolBlock(out_chs, group_size),
        )
    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """업샘플링: ConvTranspose로 크기 2배 → conv 4번."""
    def __init__(self, in_chs, out_chs, group_size):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(2 * in_chs, out_chs, 2, 2),
            GELUConvBlock(out_chs, out_chs, group_size),
            GELUConvBlock(out_chs, out_chs, group_size),
            GELUConvBlock(out_chs, out_chs, group_size),
            GELUConvBlock(out_chs, out_chs, group_size),
        )
    def forward(self, x, skip):
        return self.block(torch.cat([x, skip], dim=1))


class EmbedBlock(nn.Module):
    """임베딩 벡터를 (B, C, 1, 1) 형태로 바꿔주는 블록. Time/Context 둘 다 사용."""
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.input_dim = input_dim
        self.block = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
            nn.Unflatten(1, (emb_dim, 1, 1)),
        )
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.block(x)


class SinusoidalTimeEmbed(nn.Module):
    """타임스텝 t를 sin/cos 벡터로 인코딩."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


# ──────────────────────────────────────────────────────────
# 4. U-Net Denoiser
# ──────────────────────────────────────────────────────────
class UNet(nn.Module):
    """
    채널 흐름 (28×28 입력 기준):
      Encoder: 1 → 64 → 64 → 128  (28 → 14 → 7)
      Bottleneck: vector ↔ tensor
      Decoder: 128 → 64 → 64       (7 → 14 → 28)

    Time/Context 임베딩 주입 방식 (scale & shift):
      up = c_emb * up_input + t_emb
    """
    def __init__(self):
        super().__init__()
        down_chs = DOWN_CHS
        up_chs = down_chs[::-1]
        latent_sz = IMG_SZ // 4              # 28 // 4 = 7
        small_g, big_g = 8, 32

        # Encoder
        self.down0 = ResidualConvBlock(IMG_CH, down_chs[0], small_g)
        self.down1 = DownBlock(down_chs[0], down_chs[1], big_g)
        self.down2 = DownBlock(down_chs[1], down_chs[2], big_g)
        self.to_vec = nn.Sequential(nn.Flatten(), nn.GELU())

        # Bottleneck (vector 처리)
        flat = down_chs[2] * latent_sz ** 2
        self.dense_emb = nn.Sequential(
            nn.Linear(flat, down_chs[1]), nn.ReLU(),
            nn.Linear(down_chs[1], down_chs[1]), nn.ReLU(),
            nn.Linear(down_chs[1], flat), nn.ReLU(),
        )

        # Time / Context 임베딩
        self.sin_t = SinusoidalTimeEmbed(T_EMBED)
        self.t_emb1 = EmbedBlock(T_EMBED, up_chs[0])
        self.t_emb2 = EmbedBlock(T_EMBED, up_chs[1])
        self.c_emb1 = EmbedBlock(N_CLS, up_chs[0])
        self.c_emb2 = EmbedBlock(N_CLS, up_chs[1])

        # Decoder
        self.up0 = nn.Sequential(
            nn.Unflatten(1, (up_chs[0], latent_sz, latent_sz)),
            GELUConvBlock(up_chs[0], up_chs[0], big_g),
        )
        self.up1 = UpBlock(up_chs[0], up_chs[1], big_g)
        self.up2 = UpBlock(up_chs[1], up_chs[2], big_g)

        # Output
        self.out = nn.Sequential(
            nn.Conv2d(2 * up_chs[-1], up_chs[-1], 3, padding=1),
            nn.GroupNorm(small_g, up_chs[-1]),
            nn.ReLU(),
            nn.Conv2d(up_chs[-1], IMG_CH, 3, padding=1),
        )

    def forward(self, x, t, c, c_mask):
        d0 = self.down0(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)

        latent = self.dense_emb(self.to_vec(d2))

        # 시간 정보를 [0,1]로 정규화 → sin/cos 인코딩
        t = self.sin_t(t.float() / T)
        t1, t2 = self.t_emb1(t), self.t_emb2(t)

        # 클래스 정보에 Bernoulli 마스크 적용 후 임베딩
        c = c * c_mask
        c1, c2 = self.c_emb1(c), self.c_emb2(c)

        # 디코더에 Time/Context 주입 (scale & shift)
        u0 = self.up0(latent)
        u1 = self.up1(c1 * u0 + t1, d2)
        u2 = self.up2(c2 * u1 + t2, d1)

        return self.out(torch.cat([u2, d0], dim=1))


model = UNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
n_params = sum(p.numel() for p in model.parameters())
print(f'Model parameters: {n_params:,}')

# ──────────────────────────────────────────────────────────
# 5. CFG용 Context Mask & Loss
# ──────────────────────────────────────────────────────────
def get_context_mask(c_int, drop_prob):
    """
    클래스 정수 → one-hot 인코딩.
    Bernoulli 분포로 일부 차원을 0으로 만들어 unconditional 학습을 함께 진행.
    """
    c_hot  = F.one_hot(c_int.long(), num_classes=N_CLS).float().to(DEVICE)
    c_mask = torch.bernoulli(torch.ones_like(c_hot) - drop_prob).to(DEVICE)
    return c_hot, c_mask


def get_loss(x_0, t, c_hot, c_mask):
    """예측한 노이즈와 실제 노이즈의 MSE."""
    x_noisy, noise = q_sample(x_0, t)
    noise_pred = model(x_noisy, t, c_hot, c_mask)
    return F.mse_loss(noise, noise_pred)


# ──────────────────────────────────────────────────────────
# 6. 학습 루프
# ──────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate():
    model.eval()
    total, count = 0.0, 0
    for xb, cb in val_dl:
        xb, cb = xb.to(DEVICE), cb.to(DEVICE)
        t = torch.randint(0, T, (xb.size(0),), device=DEVICE).float()
        c_hot, c_mask = get_context_mask(cb, DROP_P)
        loss = get_loss(xb, t, c_hot, c_mask)
        total += loss.item() * xb.size(0)
        count += xb.size(0)
    model.train()
    return total / count


train_losses, val_losses = [], []

print('\n[Training Started]')
for epoch in range(1, EPOCHS + 1):
    model.train()
    ep_loss = 0.0
    for xb, cb in train_dl:
        xb, cb = xb.to(DEVICE), cb.to(DEVICE)
        t = torch.randint(0, T, (xb.size(0),), device=DEVICE).float()
        c_hot, c_mask = get_context_mask(cb, DROP_P)

        optimizer.zero_grad()
        loss = get_loss(xb, t, c_hot, c_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ep_loss += loss.item()

    scheduler.step()
    tr = ep_loss / len(train_dl)
    vl = evaluate()
    train_losses.append(tr)
    val_losses.append(vl)
    print(f'  Epoch {epoch:02d}/{EPOCHS} | Train: {tr:.5f} | Val: {vl:.5f}')


# ──────────────────────────────────────────────────────────
# 7. CFG 샘플링
#    배치를 두 배로 만들어 한 번의 forward로
#    conditional / unconditional 노이즈를 동시에 얻는다.
# ──────────────────────────────────────────────────────────
@torch.no_grad()
def sample_w(c_hot, w):
    """ε̂ = (1+w)·ε_keep - w·ε_drop"""
    n = c_hot.size(0)
    w_t = torch.tensor([w], device=DEVICE).view(1, 1, 1, 1)
    x_t = torch.randn(n, IMG_CH, IMG_SZ, IMG_SZ, device=DEVICE)

    c = c_hot.repeat(2, 1)
    c_mask = torch.ones_like(c, device=DEVICE)
    c_mask[n:] = 0.0   # 후반 절반은 unconditional

    for i in reversed(range(T)):
        t = torch.full((n,), i, device=DEVICE).float()
        x_dbl = x_t.repeat(2, 1, 1, 1)
        t_dbl = t.repeat(2)

        e = model(x_dbl, t_dbl, c, c_mask)
        e_keep, e_drop = e[:n], e[n:]
        e = (1 + w_t) * e_keep - w_t * e_drop

        x_t = reverse_q(x_t, t, e)

    return x_t.clamp(-1, 1).cpu()


@torch.no_grad()
def generate(cls_idx, n=8, w=2.0):
    model.eval()
    c_int = torch.tensor([cls_idx] * n, device=DEVICE)
    c_hot, _ = get_context_mask(c_int, drop_prob=0.0)
    return sample_w(c_hot, w)


# ──────────────────────────────────────────────────────────
# Plot 1 — Forward Diffusion 궤적
# ──────────────────────────────────────────────────────────
def plot_forward():
    sample, lbl = train_ds[7]
    x0 = sample.unsqueeze(0).to(DEVICE)
    steps = [0, 50, 100, 200, 300, T - 1]

    fig, axes = plt.subplots(1, len(steps), figsize=(13, 2.6))
    fig.suptitle(f'Forward Diffusion — {LABELS[lbl]} (t: 0 → {T-1})',
                 fontsize=11, fontweight='bold')
    for ax, step in zip(axes, steps):
        x_t, _ = q_sample(x0, torch.tensor([step], device=DEVICE))
        ax.imshow(x_t[0, 0].cpu(), cmap='gray', vmin=-1, vmax=1)
        ax.set_title(f't = {step}', fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/01_forward_diffusion.png', dpi=130, bbox_inches='tight')
    plt.close()
    print('[Saved] 01_forward_diffusion.png')

plot_forward()

# ──────────────────────────────────────────────────────────
# Plot 2 — 학습 곡선
# ──────────────────────────────────────────────────────────
def plot_training_curve():
    fig, ax = plt.subplots(figsize=(8, 4))
    ep = range(1, EPOCHS + 1)
    ax.plot(ep, train_losses, 'o-',  label='Train', color='steelblue')
    ax.plot(ep, val_losses,   's--', label='Val',   color='crimson')
    ax.set_title('Training Curve — Noise Prediction MSE', fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/02_training_curve.png', dpi=130, bbox_inches='tight')
    plt.close()
    print('[Saved] 02_training_curve.png')

plot_training_curve()

# ──────────────────────────────────────────────────────────
# Plot 3 — Reverse Diffusion 궤적
# ──────────────────────────────────────────────────────────
def plot_reverse(cls_idx=7):
    """Pure Noise → 이미지 복원 과정 시각화."""
    model.eval()
    c_int = torch.tensor([cls_idx], device=DEVICE)
    c_hot, _ = get_context_mask(c_int, drop_prob=0.0)
    c = c_hot.repeat(2, 1)
    c_mask = torch.ones_like(c, device=DEVICE); c_mask[1:] = 0.0
    w_t = torch.tensor([2.0], device=DEVICE).view(1, 1, 1, 1)
    x_t = torch.randn(1, IMG_CH, IMG_SZ, IMG_SZ, device=DEVICE)

    snap_at = sorted({T - 1, 300, 200, 100, 50, 20, 0})
    snaps = {}
    with torch.no_grad():
        for i in reversed(range(T)):
            t = torch.tensor([i], device=DEVICE).float()
            e = model(x_t.repeat(2, 1, 1, 1), t.repeat(2), c, c_mask)
            e = (1 + w_t) * e[:1] - w_t * e[1:]
            x_t = reverse_q(x_t, t, e)
            if i in snap_at:
                snaps[i] = x_t.clamp(-1, 1)[0, 0].cpu().numpy()

    steps = sorted(snaps.keys(), reverse=True)
    fig, axes = plt.subplots(1, len(steps), figsize=(13, 2.6))
    fig.suptitle(f'Reverse Diffusion — {LABELS[cls_idx]}  (CFG w=2.0)',
                 fontsize=11, fontweight='bold')
    for ax, step in zip(axes, steps):
        ax.imshow(snaps[step], cmap='gray', vmin=-1, vmax=1)
        ax.set_title(f't = {step}', fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/03_reverse_trajectory.png', dpi=130, bbox_inches='tight')
    plt.close()
    print('[Saved] 03_reverse_trajectory.png')

plot_reverse(cls_idx=7)

# ──────────────────────────────────────────────────────────
# Plot 4 — 클래스별 생성 결과
# ──────────────────────────────────────────────────────────
def plot_generated_grid():
    n_samples = 5
    fig, axes = plt.subplots(N_CLS, n_samples, figsize=(9, 16))
    fig.suptitle('Generated Samples — All Classes (CFG w=2.0)',
                 fontsize=12, fontweight='bold')
    for ci, lbl in enumerate(LABELS):
        imgs = generate(ci, n=n_samples, w=2.0)
        for j in range(n_samples):
            ax = axes[ci, j]
            ax.imshow(imgs[j, 0].numpy(), cmap='gray', vmin=-1, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if j == 0:
                ax.set_ylabel(lbl, rotation=0, ha='right', va='center',
                              fontsize=10, fontweight='bold', labelpad=8)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/04_generated_samples.png', dpi=130, bbox_inches='tight')
    plt.close()
    print('[Saved] 04_generated_samples.png')

plot_generated_grid()

# ──────────────────────────────────────────────────────────
# Plot 5 — CFG Weight 비교
# ──────────────────────────────────────────────────────────
def plot_cfg_comparison(cls_idx=0):
    """
    w 값에 따라 클래스 특성이 어떻게 강해지는지 비교.
    w=0이면 unconditional, w가 커질수록 클래스 특성이 강조.
    """
    w_vals = [-1.0, 0.0, 1.0, 2.0, 4.0]
    n_col = 4
    fig, axes = plt.subplots(len(w_vals), n_col, figsize=(9, 10))
    fig.suptitle(f'CFG Weight Comparison — {LABELS[cls_idx]}',
                 fontsize=12, fontweight='bold')
    for row, w in enumerate(w_vals):
        imgs = generate(cls_idx, n=n_col, w=w)
        for col in range(n_col):
            ax = axes[row, col]
            ax.imshow(imgs[col, 0].numpy(), cmap='gray', vmin=-1, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col == 0:
                ax.set_ylabel(f'w = {w:+.1f}', rotation=0, ha='right', va='center',
                              fontsize=11, fontweight='bold', labelpad=8)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/05_cfg_weight_comparison.png', dpi=130, bbox_inches='tight')
    plt.close()
    print('[Saved] 05_cfg_weight_comparison.png')

plot_cfg_comparison(cls_idx=0)

# ──────────────────────────────────────────────────────────
# 최종 요약
# ──────────────────────────────────────────────────────────
print('\n' + '=' * 50)
print('  Run Summary')
print('=' * 50)
print(f'  Device          : {DEVICE}')
print(f'  Model params    : {n_params:,}')
print(f'  Diffusion steps : {T}')
print(f'  Epochs trained  : {EPOCHS}')
print(f'  Final Train Loss: {train_losses[-1]:.5f}')
print(f'  Final Val Loss  : {val_losses[-1]:.5f}')
print(f'  Saved plots     : {SAVE_DIR}/ (5 files)')
print('=' * 50)
