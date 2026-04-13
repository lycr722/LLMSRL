# losses.py
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEAndMultiLabelMarginLoss(nn.Module):
    """
    混合损失 = λ * BCEWithLogits + (1-λ) * MultiLabelMarginLoss
    适配你当前的 [B, T, V] 形状（时间步 T 的多标签）。
    - BCEWithLogits 用于概率校准与绝对值逼近；
    - MultiLabelMarginLoss 用于正-负类别之间的排序间隔（排名更靠前）。
    """

    def __init__(self, vocab_size: int, bce_weight: float = 0.7, reduction: str = "mean",
                 pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        assert 0.0 <= bce_weight <= 1.0
        self.vocab_size = vocab_size  # V
        self.bce_weight = bce_weight
        self.reduction = reduction
        # pos_weight 可以是 1D tensor[ V ]，用于应对类别不平衡
        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor(pos_weight, dtype=torch.float)
        self.register_buffer("pos_weight_buf", pos_weight if pos_weight is not None else None)
        self.margin_loss = nn.MultiLabelMarginLoss(reduction="none")  # 返回 [N]，N为样本数

    @torch.no_grad()
    def _build_target_indices(self, targets_2d: torch.Tensor) -> torch.Tensor:
        """
        将二值 multi-hot 标签 [N, V] 转为 MultiLabelMarginLoss 需要的
        索引矩阵 [N, V]，正类索引填入前面，剩余位置填 -1。
        """
        device = targets_2d.device
        N, V = targets_2d.shape
        target_idx = torch.full((N, V), -1, dtype=torch.long, device=device)
        # 逐样本填充正类索引（V=131 量级，循环开销可接受）
        for i in range(N):
            pos = torch.nonzero(targets_2d[i] > 0, as_tuple=False).squeeze(-1)
            if pos.numel() > 0:
                k = min(pos.numel(), V)
                target_idx[i, :k] = pos[:k]
        return target_idx

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [B, T, V]  (未经 sigmoid)
        targets: [B, T, V]  (0/1)
        """
        B, T, V = logits.shape
        assert V == self.vocab_size, "vocab_size 与 logits 最后维不一致"

        # --- BCEWithLogits ---
        if self.pos_weight_buf is not None:
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weight_buf.to(logits.device), reduction="none"
            )  # [B, T, V]
        else:
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # [B, T, V]
        # 按类别平均，得到每个时间步的 BCE
        bce = bce.mean(dim=-1)  # [B, T]

        # --- MultiLabelMarginLoss（排序间隔） ---
        logits_2d = logits.reshape(-1, V)  # [B*T, V]
        targets_2d = (targets > 0.5).reshape(-1, V)  # [B*T, V] -> bool
        target_idx = self._build_target_indices(targets_2d)  # [B*T, V]
        mlm = self.margin_loss(logits_2d, target_idx)  # [B*T]
        mlm = mlm.view(B, T)  # [B, T]

        # --- 加权融合 ---
        mixed = self.bce_weight * bce + (1.0 - self.bce_weight) * mlm  # [B, T]

        if self.reduction == "mean":
            return mixed.mean()
        elif self.reduction == "sum":
            return mixed.sum()
        else:  # "none"
            return mixed  # [B, T]


class BCEAndMarginLoss(nn.Module):
    def __init__(self, margin_weight: float = 1.0, bce_pos_weight: Optional[torch.Tensor] = None):
        """
        混合 BCE 和 Multi-Label Margin Loss.

        Args:
            margin_weight (float): 多标签边际损失的权重.
            bce_pos_weight (torch.Tensor, optional): 用于BCE损失的类别权重，可以帮助处理类别不平衡.
        """
        super().__init__()
        self.margin_weight = margin_weight
        self.bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight)
        self.margin_loss_fn = nn.MultiLabelMarginLoss()

    def forward(self, logits: torch.Tensor, targets_multihot: torch.Tensor) -> torch.Tensor:
        """
        计算混合损失.

        Args:
            logits (torch.Tensor): 模型的原始输出, 形状为 [batch_size, num_classes].
            targets_multihot (torch.Tensor): multi-hot 编码的真实标签, 形状为 [batch_size, num_classes].

        Returns:
            torch.Tensor: 计算得到的总损失.
        """
        # 1. 计算标准的 BCE Loss
        bce_loss = self.bce_loss_fn(logits, targets_multihot)

        # 2. 转换标签格式以适应 MultiLabelMarginLoss
        #    将 multi-hot 格式 [1, 0, 1, 0] 转换为 index 格式 [0, 2, -1, -1]
        targets_indices = []
        max_pos_labels = 0
        for i in range(targets_multihot.size(0)):
            # 找到正样本的索引
            indices = (targets_multihot[i] == 1).nonzero(as_tuple=False).squeeze(-1)
            targets_indices.append(indices)
            if len(indices) > max_pos_labels:
                max_pos_labels = len(indices)

        # 用 -1 填充，使其成为一个矩形张量
        targets_for_margin = -torch.ones(targets_multihot.size(0), max_pos_labels, dtype=torch.long,
                                         device=logits.device)
        for i, indices in enumerate(targets_indices):
            if len(indices) > 0:
                targets_for_margin[i, :len(indices)] = indices

        # 3. 计算 Multi-Label Margin Loss
        margin_loss = self.margin_loss_fn(logits, targets_for_margin)

        # 4. 返回加权总损失
        total_loss = bce_loss + self.margin_weight * margin_loss

        return total_loss


class DiceBCELoss(nn.Module):
    """
    DiceBCELoss类结合了Dice损失和二元交叉熵（BCE）损失。
    这种混合损失函数旨在利用BCE的训练稳定性，同时通过Dice损失直接优化
    与Jaccard分数高度相关的集合相似性度量。
    """

    def __init__(self, from_logits: bool = True, bce_weight: float = 0.5, smooth: float = 1e-7):
        """
        初始化函数。

        参数:
            from_logits (bool): 如果为True，则假定输入是原始的logits，将对其应用sigmoid激活。
                                默认为True，因为这通常是PyTorch中更数值稳定的做法。
            bce_weight (float): BCE损失分量的权重。Dice损失的权重将是 (1 - bce_weight)。
                                默认为0.5，给予两者相等的权重。
            smooth (float): 用于Dice损失计算中的平滑常数，以避免除以零的错误。
                            根据研究建议，应设置为一个非常小的值。默认为1e-7。
        """
        super(DiceBCELoss, self).__init__()
        self.from_logits = from_logits
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            y_pred (torch.Tensor): 模型的预测输出 (logits或概率)。形状:。
            y_true (torch.Tensor): 真实的标签。形状:。

        返回:
            torch.Tensor: 计算出的混合损失值。
        """
        # --- BCE损失部分 ---
        if self.from_logits:
            # F.binary_cross_entropy_with_logits 内部会处理sigmoid，更稳定
            bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='mean')
            y_pred_prob = torch.sigmoid(y_pred)
            # print(f"二元交叉熵 {bce_loss}")
        else:
            bce_loss = F.binary_cross_entropy(y_pred, y_true, reduction='mean')
            y_pred_prob = y_pred

        # --- Dice损失部分 ---
        # 将张量展平以便计算
        y_pred_flat = y_pred_prob.view(-1)
        y_true_flat = y_true.view(-1)

        # 计算交集
        intersection = (y_pred_flat * y_true_flat).sum()

        # 计算Dice系数
        dice_score = (2. * intersection + self.smooth) / (y_pred_flat.sum() + y_true_flat.sum() + self.smooth)

        # Dice损失是 1 - Dice系数
        dice_loss = 1 - dice_score
        # print(f"Dice损失 {dice_loss}")

        # --- 结合损失 ---
        # 根据bce_weight加权求和
        combined_loss = self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss
        # print(f"融合损失 {combined_loss}")

        return combined_loss