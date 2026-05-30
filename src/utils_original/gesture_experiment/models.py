from .common import *
from .config import *


class AugmentedGestureDataset(Dataset):
    def __init__(self, X_tensor, y_tensor, is_training=False, use_aug=True):
        self.X = X_tensor
        self.y = y_tensor
        self.is_training = is_training
        self.use_aug = use_aug

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        label = self.y[idx]

        if self.is_training and self.use_aug:
            # Translation: 整只手整体平移
            if torch.rand(1).item() > 0.5:
                shift = torch.empty(1, 3).uniform_(-0.05, 0.05)
                x += shift

            # Scaling: 以坐标原点为中心缩放
            if torch.rand(1).item() > 0.5:
                scale = torch.empty(1).uniform_(0.8, 1.2)
                x *= scale

            # Landmark jitter
            if torch.rand(1).item() > 0.5:
                noise = torch.randn_like(x) * 0.005
                x += noise

        return x, label


# ============================================================
# 3. Models
# ============================================================

class BaselineMLP(nn.Module):
    def __init__(self, num_classes: int, cfg: ExperimentConfig):
        super().__init__()

        self.num_landmarks = cfg.num_landmarks
        self.coord_dim = cfg.coord_dim
        input_features = cfg.num_landmarks * cfg.coord_dim

        self.fc = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        logits = self.fc(x)
        return logits, None, None


def build_hand_graph_adjacency(num_landmarks: int):
    """MediaPipe hand kinematic graph with self loops for GCN baselines."""
    chains = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20],
    ]

    adjacency = torch.eye(num_landmarks, dtype=torch.float32)
    for chain in chains:
        for src, dst in zip(chain[:-1], chain[1:]):
            if src >= num_landmarks or dst >= num_landmarks:
                continue
            adjacency[src, dst] = 1.0
            adjacency[dst, src] = 1.0

    degree = adjacency.sum(dim=1).clamp_min(1.0)
    inv_sqrt_degree = torch.diag(torch.pow(degree, -0.5))
    return inv_sqrt_degree @ adjacency @ inv_sqrt_degree


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency):
        propagated = torch.einsum("ij,bjf->bif", adjacency, x)
        return self.proj(propagated)


class BaselineGCN(nn.Module):
    def __init__(self, num_classes: int, cfg: ExperimentConfig):
        super().__init__()

        self.num_landmarks = cfg.num_landmarks
        self.coord_dim = cfg.coord_dim
        self.register_buffer(
            "adjacency",
            build_hand_graph_adjacency(cfg.num_landmarks),
        )

        self.gcn1 = GraphConvolution(cfg.coord_dim, cfg.embed_dim)
        self.gcn2 = GraphConvolution(cfg.embed_dim, cfg.embed_dim)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.embed_dim * cfg.num_landmarks, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = F.relu(self.gcn1(x, self.adjacency))
        x = self.dropout(x)
        x = F.relu(self.gcn2(x, self.adjacency))
        logits = self.classifier(x.reshape(x.shape[0], -1))
        return logits, None, None



class BaselinePointNet1D(nn.Module):
    """
    PointNet-style / 1D-CNN landmark classifier.

    It treats each hand landmark as a point token with 3 coordinate channels and
    applies shared 1x1 convolutions followed by global max pooling. This is a
    common lightweight baseline for unordered or weakly ordered landmark sets.
    """
    def __init__(self, num_classes: int, cfg: ExperimentConfig):
        super().__init__()

        self.num_landmarks = cfg.num_landmarks
        self.coord_dim = cfg.coord_dim

        self.point_mlp = nn.Sequential(
            nn.Conv1d(cfg.coord_dim, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, cfg.embed_dim, kernel_size=1),
            nn.BatchNorm1d(cfg.embed_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(cfg.embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: [B, 21, 3] -> [B, 3, 21]
        x = x.transpose(1, 2)
        x = self.point_mlp(x)
        x = torch.max(x, dim=2).values
        logits = self.classifier(x)
        return logits, None, None


class BaselineTransformerLite(nn.Module):
    """
    Lightweight Transformer baseline for landmark tokens.

    Unlike the proposed model, this baseline uses a standard TransformerEncoder
    and does not include the reconstruction/anatomical regularization branch.
    """
    def __init__(self, num_classes: int, cfg: ExperimentConfig):
        super().__init__()

        self.num_landmarks = cfg.num_landmarks
        self.coord_dim = cfg.coord_dim
        self.embed_dim = cfg.embed_dim

        self.coord_embed = nn.Linear(cfg.coord_dim, cfg.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.num_landmarks, cfg.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.embed_dim * 2,
            dropout=0.1,
            batch_first=True,
            activation="relu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.classifier = nn.Sequential(
            nn.Linear(cfg.embed_dim * cfg.num_landmarks, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.coord_embed(x) + self.pos_embed
        x = self.encoder(x)
        logits = self.classifier(x.reshape(x.shape[0], -1))
        return logits, None, None


class GestureAttentionModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.embed_dim
        self.num_landmarks = cfg.num_landmarks
        self.coord_dim = cfg.coord_dim

        self.coord_embed = nn.Linear(input_dim, self.embed_dim)

        if self.cfg.use_positional_encoding:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, cfg.num_landmarks, self.embed_dim)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        if self.cfg.use_attention:
            self.attn = nn.MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=cfg.num_heads,
                batch_first=True,
            )

        self.fc = nn.Sequential(
            nn.Linear(self.embed_dim * cfg.num_landmarks, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        # Reconstruction branch is used only when anatomical regularization is enabled.
        if self.cfg.use_anatomical_loss:
            self.reconstruction_head = nn.Linear(self.embed_dim, input_dim)

    def forward(self, x):
        embedded = self.coord_embed(x)

        if self.cfg.use_positional_encoding:
            embedded = embedded + self.pos_embed

        attn_weights = None
        if self.cfg.use_attention:
            attn_out, attn_weights = self.attn(embedded, embedded, embedded)
            if self.cfg.use_residual:
                fused = embedded + attn_out
            else:
                fused = attn_out
        else:
            fused = embedded

        flattened = fused.reshape(x.shape[0], -1)
        logits = self.fc(flattened)

        reconstructed = None
        if self.cfg.use_anatomical_loss:
            reconstructed = self.reconstruction_head(fused)

        return logits, attn_weights, reconstructed


def build_model(cfg: ExperimentConfig, num_classes: int):
    if cfg.model_type == "mlp":
        return BaselineMLP(num_classes=num_classes, cfg=cfg)
    if cfg.model_type == "pointnet":
        return BaselinePointNet1D(num_classes=num_classes, cfg=cfg)
    if cfg.model_type == "gcn":
        return BaselineGCN(num_classes=num_classes, cfg=cfg)
    if cfg.model_type == "transformer":
        return BaselineTransformerLite(num_classes=num_classes, cfg=cfg)
    if cfg.model_type == "svm":
        raise ValueError("SVM is trained by run_svm_experiment, not build_model.")
    if cfg.model_type == "attn":
        return GestureAttentionModel(
            input_dim=cfg.coord_dim,
            num_classes=num_classes,
            cfg=cfg,
        )

    raise ValueError(f"Unknown model_type: {cfg.model_type}")


# ============================================================
# 4. Anatomical Regularization
# ============================================================

FINGER_CHAINS = [
    [0, 1, 2, 3, 4],       # thumb
    [0, 5, 6, 7, 8],       # index
    [0, 9, 10, 11, 12],    # middle
    [0, 13, 14, 15, 16],   # ring
    [0, 17, 18, 19, 20],   # pinky
]


def inverse_standardize_landmarks(x_tensor, scaler):
    """
    x_tensor: [B, 21, 3] for single-hand mode.
    scaler: sklearn StandardScaler fitted on flattened 63-dim training data.
    """
    if scaler is None:
        return x_tensor

    bsz = x_tensor.shape[0]
    landmark_shape = x_tensor.shape[1:]  # [21, 3]
    flat = x_tensor.reshape(bsz, -1)

    mean = torch.as_tensor(scaler.mean_, dtype=x_tensor.dtype, device=x_tensor.device)
    scale = torch.as_tensor(scaler.scale_, dtype=x_tensor.dtype, device=x_tensor.device)

    raw = flat * scale + mean
    return raw.reshape(bsz, *landmark_shape)


def anatomical_regularization(landmarks_raw, reference_raw=None):
    """
    Single-hand anatomical regularization.

    landmarks_raw: [B, 21, 3]
    reference_raw: [B, 21, 3]
    """
    eps = 1e-6
    hand = landmarks_raw

    if reference_raw is not None:
        valid_mask = (reference_raw.abs().sum(dim=(1, 2)) > 1e-6).float()
    else:
        valid_mask = (hand.abs().sum(dim=(1, 2)) > 1e-6).float()

    sample_loss = hand.new_zeros(hand.shape[0])

    # 1) Finger length-ratio smoothness.
    for chain in FINGER_CHAINS:
        seg_lengths = []
        for a, b in zip(chain[:-1], chain[1:]):
            seg_lengths.append(torch.norm(hand[:, a, :] - hand[:, b, :], dim=-1) + eps)

        seg_lengths = torch.stack(seg_lengths, dim=1)  # [B, 4]
        ratios = seg_lengths[:, 1:] / (seg_lengths[:, :-1] + eps)

        ratio_loss = (
            F.relu(ratios - 3.0) ** 2
            + F.relu(0.25 - ratios) ** 2
        ).mean(dim=1)

        sample_loss = sample_loss + ratio_loss

    # 2) Palm planarity prior.
    p0 = hand[:, 0, :]
    p5 = hand[:, 5, :]
    p9 = hand[:, 9, :]
    p13 = hand[:, 13, :]
    p17 = hand[:, 17, :]

    normal = torch.cross(p9 - p0, p17 - p0, dim=-1)
    normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + eps)

    dist5 = torch.abs(torch.sum((p5 - p0) * normal, dim=-1))
    dist13 = torch.abs(torch.sum((p13 - p0) * normal, dim=-1))
    plane_loss = dist5 ** 2 + dist13 ** 2

    sample_loss = sample_loss + plane_loss

    return (sample_loss * valid_mask).sum() / (valid_mask.sum() + eps)


def compute_loss(outputs, labels, reconstructed, inputs, cfg, scaler):
    ce_loss = F.cross_entropy(outputs, labels)

    if not cfg.use_anatomical_loss or reconstructed is None:
        return ce_loss, {
            "ce_loss": ce_loss.item(),
            "recon_loss": 0.0,
            "anatomical_loss": 0.0,
        }

    reconstruction_loss = F.mse_loss(reconstructed, inputs)

    inputs_raw = inverse_standardize_landmarks(inputs, scaler)
    reconstructed_raw = inverse_standardize_landmarks(reconstructed, scaler)

    anatomical_loss = anatomical_regularization(
        landmarks_raw=reconstructed_raw,
        reference_raw=inputs_raw,
    )

    total = (
        ce_loss
        + cfg.reconstruction_weight * reconstruction_loss
        + cfg.anatomical_weight * anatomical_loss
    )

    return total, {
        "ce_loss": ce_loss.item(),
        "recon_loss": reconstruction_loss.item(),
        "anatomical_loss": anatomical_loss.item(),
    }
