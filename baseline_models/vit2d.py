import torch
from torch import nn
from einops import rearrange
import pytorch_lightning as pl
import torch.nn.functional as F
# -------------------------
# 2D sin/cos 位置编码（无参数，可适配任意 H,W，只要是 patch 的整数倍）
# -------------------------
def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32, device=None):
    assert dim % 4 == 0, "dim 必须能被 4 整除"
    y, x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij"
    )
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]

    pe = torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)  # (h*w, dim)
    return pe.to(dtype)

# -------------------------
# 基本 Transformer 积木（Pre-LN）
# -------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.qkv = nn.Linear(dim, inner * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(inner, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = self.norm(x)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, -1).permute(2, 0, 3, 1, 4)  # 3,B,H,N,D
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, attn_drop=0.0, proj_drop=0.0, mlp_drop=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads, dim_head, attn_drop, proj_drop),
                FeedForward(dim, mlp_dim, drop=mlp_drop)
            ]) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)
        return self.norm(x)

# -------------------------
# ViT 编码 + 线性解码回像素（图到图回归）
# -------------------------
class ViTImage2Image(pl.LightningModule):
    """
    输入:  B x in_ch x H x W   （H,W 需为 patch_size 的整数倍）
    输出:  B x out_ch x H x W  （这里 out_ch=1）
    """
    def __init__(
        self,
        in_ch=3,
        out_ch=1,
        patch_size=16,
        dim=24,  # 256
        depth=1,  # 8
        heads=3,  # 8
        dim_head=8,  # 64
        mlp_dim=48,  # 512
        attn_drop=0.0,
        proj_drop=0.0,
        mlp_drop=0.0,
        **kwargs,
    ):  
        super().__init__()
        self.save_hyperparameters()
        print(self.hparams)

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.num_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'\nThe model has {self.num_params} params \n')

        self.patch = patch_size
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dim = dim

        # Patch Embedding: Conv2d 等价于线性投影 + 切块，且更快
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(dim)

        self.encoder = Transformer(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head,
            mlp_dim=mlp_dim, attn_drop=attn_drop, proj_drop=proj_drop, mlp_drop=mlp_drop
        )

        # 将 token 表示还原成每个 patch 的像素：dim -> (out_ch * patch * patch)
        self.patch_decoder = nn.Linear(dim, out_ch * patch_size * patch_size)

    def forward(self, x):
        """
        x: (B, in_ch, H, W)
        return: (B, out_ch, H, W)
        """
        B, _, H, W = x.shape
        p = self.patch

        # 计算需要补到的尺寸：补到 p 的整数倍
        pad_h = (p - (H % p)) % p
        pad_w = (p - (W % p)) % p

        if pad_h != 0 or pad_w != 0:
            # pad 顺序: (left, right, top, bottom)
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # 经过 patch-embed
        x = self.patch_embed(x)            # (B, dim, H', W')  其中 H'=ceil(H/p), W'=ceil(W/p)
        h, w = x.shape[-2], x.shape[-1]

        # token 化
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)

        # 位置编码（按补后的 h,w 生成）
        pos = posemb_sincos_2d(h, w, self.dim, device=x.device, dtype=x.dtype)  # (h*w, dim)
        x = x + pos.unsqueeze(0)

        # Transformer
        x = self.encoder(x)

        # 每个 token 解码回 patch 像素
        x = self.patch_decoder(x)          # (B, h*w, out_ch*p*p)

        # 拼回图像
        x = rearrange(
            x, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
            h=h, w=w, c=self.out_ch, p1=p, p2=p
        )

        # 如果做过 padding，这里裁回原始 H,W
        if pad_h != 0 or pad_w != 0:
            x = x[:, :, :H, :W]

        return x
    

    def training_step(self, batch, batch_idx):
        input, gt = batch
        # gt = gt.squeeze()
        background = input[:,2:3,:,:]
        pred_values = background + self(input)
        loss = F.mse_loss(pred_values, gt, reduction='mean')
        self.log("train_loss", loss, on_step=True, on_epoch=True,prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input, gt = batch
        # gt = gt.squeeze()
        background = input[:,2:3,:,:]
        pred_values = background + self(input)
        loss = F.mse_loss(pred_values, gt, reduction='mean')
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)



# model = ViTImage2Image(
#     in_ch=3, out_ch=1,
#     patch_size=16,   # 8/16/32 都可以，越小越精细但算力更大
#     dim=256, depth=8, heads=8, dim_head=64, mlp_dim=512,
#     attn_drop=0.0, proj_drop=0.0, mlp_drop=0.0
# )

# x = torch.randn(2, 3, 224, 224)      # H,W 需为 patch 的整数倍
# y = model(x)                         # (2, 1, 224, 224)
# print(y.shape)
# import numpy as np  # 你上面统计参数量的地方需要这个

# model = ViTImage2Image(
#     in_ch=3, out_ch=1,
#     patch_size=16,
#     dim=24,           # ↓↓↓ 关键三处
#     depth=1,
#     heads=3, dim_head=8,
#     mlp_dim=48,
#     attn_drop=0.0, proj_drop=0.0, mlp_drop=0.0,
#     lr=1e-4           # 你的 configure_optimizers 需要这个字段
# )
