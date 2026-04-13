import os
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from pyhealth.models import BaseModel
from pyhealth.datasets import SampleEHRDataset
from models.layers import LabelAttention
from models.losses import DiceBCELoss
from pyhealth.metrics import ddi_rate_score


# --- T3Time 组件：自适应多头对齐 ---
class AdaptiveMultiHeadAlignment(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            nn.MultiheadAttention(dim, 1, batch_first=True) for _ in range(num_heads)
        ])
        self.head_gating = nn.Sequential(
            nn.Linear(dim * num_heads, 128),
            nn.ReLU(),
            nn.Linear(128, num_heads),
            nn.Softmax(dim=-1)
        )

    def forward(self, query, key, value):
        head_outputs = [head(query, key, value)[0] for head in self.heads]
        concat_out = torch.cat(head_outputs, dim=-1)
        weights = self.head_gating(concat_out)
        final_out = torch.zeros_like(head_outputs[0])
        for i in range(self.num_heads):
            final_out += weights[:, :, i].unsqueeze(-1) * head_outputs[i]
        return final_out


# --- ET-Fake 组件：层次性融合 ---
class HierarchicalModalFusion(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.cam1 = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gfm1 = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.cam2 = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gfm2 = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_temp, x_sem, x_struct):
        # 阶段 1: Temporal + Semantic
        attn1, _ = self.cam1(x_temp, x_sem, x_sem)
        gate1 = self.gfm1(torch.cat([x_temp, attn1], dim=-1))
        f_mid = self.norm(gate1 * attn1 + (1 - gate1) * x_temp)
        # 阶段 2: f_mid + Structural
        attn2, _ = self.cam2(f_mid, x_struct, x_struct)
        gate2 = self.gfm2(torch.cat([f_mid, attn2], dim=-1))
        return self.norm(gate2 * attn2 + (1 - gate2) * f_mid)


class LAMRec(BaseModel):
    def __init__(self, dataset: SampleEHRDataset, embedding_dim: int = 512, bce_weight: float = 0.5, **kwargs):
        super(LAMRec, self).__init__(
            dataset=dataset,
            feature_keys=["conditions", "procedures", "drugs_hist"],
            label_key="drugs",
            mode="multilabel",
        )
        self.embedding_dim = embedding_dim
        self.feat_tokenizers = self.get_feature_tokenizers()
        self.drug_tokenizer = self.get_label_tokenizer()
        self.device_type = kwargs.get("device", "cuda:0")

        # 1. 基础 Embedding (Temporal)
        self.cond_embeddings = nn.Embedding(self.feat_tokenizers['conditions'].get_vocabulary_size(), embedding_dim,
                                            padding_idx=0)
        self.proc_embeddings = nn.Embedding(self.feat_tokenizers['procedures'].get_vocabulary_size(), embedding_dim,
                                            padding_idx=0)
        self.drug_embeddings = nn.Embedding(self.drug_tokenizer.get_vocabulary_size(), embedding_dim, padding_idx=0)

        # 2. 外部知识投影
        self.sem_proj = nn.Linear(768, 512)
        self.mol_proj = nn.Linear(768, 512)

        # 3. 三模态融合模块
        self.hier_fusion = HierarchicalModalFusion(dim=512)
        self.adaptive_align = AdaptiveMultiHeadAlignment(dim=512)
        self.gamma = nn.Parameter(torch.ones(512) * 0.5)

        # 4. 加载预计算数据 (请确保路径正确)
        self.pre_cond = self.load_dict(r"D:\Code\LAMRec-RAGBK\models\MIMIC3_DIAGNOSES_Embeddings_512_ID.csv",
                                       "ICD9_CODE", is_list=True)
        self.pre_proc = self.load_dict(r"D:\Code\LAMRec-RAGBK\models\MIMIC3_PROCEDURES_Embeddings_512_ID.csv",
                                       "ICD9_CODE", is_list=True)
        # self.pre_sem = self.load_dict(r"D:\Code\LAMRec-RAGBK\models\MIMIC3_Drug_Embeddings_768_ID.csv", "code",
        #                               is_list=True)
        self.pre_mol = self.load_dict(r"D:\Code\LAMRec-RAGBK\models\aligned_drugs_and_smiles_features_768.csv", "ATC",
                                      is_list=False)

        # 5. 后端
        self.bi_gru = nn.GRU(embedding_dim * 3, embedding_dim, batch_first=True, bidirectional=True)
        self.label_attn = LabelAttention(embedding_dim * 2, embedding_dim, self.drug_tokenizer.get_vocabulary_size())
        self.loss_fn = DiceBCELoss(from_logits=True, bce_weight=bce_weight)
        self.ddi_adj = torch.tensor(
            pd.read_csv(r"D:\Code\LAMRec-RAGBK\models\ddi_adj_final_131.csv", header=None).values, dtype=torch.float32)
        self.alpha = kwargs.get("alpha", 0.07)

    def load_dict(self, path, key_col, is_list=True):
        df = pd.read_csv(path)
        d = {}
        for _, row in df.iterrows():
            val_str = row["Embedding"] if "Embedding" in row else row["aggregated_vector"]
            val = ast.literal_eval(val_str) if is_list else [float(x) for x in val_str.split()]
            d[str(row[key_col])] = nn.Parameter(torch.tensor(val, dtype=torch.float), requires_grad=True)
        return nn.ParameterDict(d)

    def get_k_emb(self, batch_data, k_dict, dim):
        res = []
        for patient in batch_data:
            flat = [c for v in patient for c in (v if isinstance(v, list) else [v]) if c != "<pad>"]
            codes = list(set(flat))
            vecs = [k_dict[c] if c in k_dict else torch.zeros(dim).to(self.device) for c in codes]
            res.append(torch.stack(vecs).mean(0) if vecs else torch.zeros(dim).to(self.device))
        return torch.stack(res).unsqueeze(1)  # [B, 1, D]

    def forward(self, conditions, procedures, drugs_hist, drugs, **kwargs):
        # --- 1. Temporal Branch ---
        c_ids = torch.tensor(self.feat_tokenizers["conditions"].batch_encode_3d(conditions), device=self.device)
        p_ids = torch.tensor(self.feat_tokenizers["procedures"].batch_encode_3d(procedures), device=self.device)

        # 【修复逻辑】：手动编码并填充，避免 Tokenizer 找不到 <pad>
        drugs_indices_jagged = []
        for patient_visits in drugs_hist:
            # 平铺当前病人的所有历史药物
            flat_codes = [c for v in patient_visits for c in (v if isinstance(v, list) else [v]) if c != "<pad>"]
            # 转换为索引，忽略未知 token 或映射到 0
            indices = self.drug_tokenizer.convert_tokens_to_indices(flat_codes)
            drugs_indices_jagged.append(indices)

        # 找到当前 Batch 的最大长度用于填充
        max_len = max(len(x) for x in drugs_indices_jagged) if drugs_indices_jagged else 1
        padded_h_ids = [x + [0] * (max_len - len(x)) for x in drugs_indices_jagged]
        h_ids = torch.tensor(padded_h_ids, dtype=torch.long, device=self.device)

        # 提取 Embedding
        x_t_c = masked_mean(self.cond_embeddings(c_ids), c_ids, 0)
        x_t_p = masked_mean(self.proc_embeddings(p_ids), p_ids, 0)

        # 处理历史药物 Embedding：如果全为空则给零向量
        h_emb = self.drug_embeddings(h_ids)
        x_t_h = h_emb.sum(dim=1, keepdim=True).expand(-1, x_t_c.size(1), -1)

        # 三模态基准特征
        x_temporal = (x_t_c + x_t_p + x_t_h) / 3

        # --- 2. Semantic Branch ---
        s_c = self.get_k_emb(conditions, self.pre_cond, 512)
        s_p = self.get_k_emb(procedures, self.pre_proc, 512)
        s_d = self.sem_proj(self.get_k_emb(drugs_hist, self.pre_sem, 768))
        x_semantic = ((s_c + s_p + s_d) / 3).expand(-1, x_temporal.size(1), -1)

        # --- 3. Structural Branch ---
        x_struct = self.mol_proj(self.get_k_emb(drugs_hist, self.pre_mol, 768)).expand(-1, x_temporal.size(1), -1)

        # --- 4. Hierarchical Fusion ---
        f_h = self.hier_fusion(x_temporal, x_semantic, x_struct)
        f_a = self.adaptive_align(f_h, x_semantic, x_semantic)
        combined = self.gamma * f_a + (1 - self.gamma) * f_h

        # --- 5. Predict ---
        # 保持输入维度为 embedding_dim * 3
        seq_in = torch.cat([combined, combined, x_t_h], dim=-1)
        gru_out, _ = self.bi_gru(seq_in)
        logits = self.label_attn(gru_out)

        y_true = self.prepare_labels(drugs, self.drug_tokenizer)
        loss = self.loss_fn(logits, y_true)

        # DDI 惩罚
        y_prob = torch.sigmoid(logits)
        if ddi_rate_score([np.where(s >= 0.5)[0] for s in y_prob.detach().cpu().numpy()],
                          self.ddi_adj.cpu().numpy()) >= 0.06:
            loss += self.alpha * (
                        torch.sum((y_prob.T @ y_prob).mul(self.ddi_adj.to(self.device))) / self.ddi_adj.shape[0] ** 2)

        return {"loss": loss, "y_prob": y_prob, "y_true": y_true}


def masked_mean(embed, ids, padding_idx):
    mask = (ids != padding_idx).float()
    return (embed * mask.unsqueeze(-1)).sum(dim=2) / mask.sum(dim=2, keepdim=True).clamp_min(1.0)