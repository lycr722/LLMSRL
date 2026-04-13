import os
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pyhealth.models import BaseModel
from pyhealth.datasets import SampleEHRDataset
from models.layers import LabelAttention
from models.losses import DiceBCELoss
from pyhealth.metrics import ddi_rate_score

# --- T3Time 组件：自适应多头对齐 (保持其强大的动态权重能力) ---
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

class HierarchicalModalFusion(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.cam1 = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        # 改进：引入两个投影层，先对齐分布再融合
        self.proj_sem = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        self.gate1 = nn.Linear(dim * 2, dim)

        self.cam2 = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.proj_struct = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        self.gate2 = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_temp, x_sem, x_struct):
        # 阶段 1: 语义增强 (残差比例调整)
        sem_aligned = self.proj_sem(x_sem)
        attn1, _ = self.cam1(x_temp, sem_aligned, sem_aligned)
        g1 = torch.sigmoid(self.gate1(torch.cat([x_temp, attn1], dim=-1)))
        f_mid = self.norm(x_temp + g1 * attn1)

        # 阶段 2: 分子纠偏
        struct_aligned = self.proj_struct(x_struct)
        attn2, _ = self.cam2(f_mid, struct_aligned, struct_aligned)
        g2 = torch.sigmoid(self.gate2(torch.cat([f_mid, attn2], dim=-1)))
        f_final = self.norm(f_mid + g2 * attn2)
        return f_final


class LAMRec(BaseModel):
    def __init__(self, dataset, embedding_dim=512, bce_weight=0.5, **kwargs):
        super(LAMRec, self).__init__(dataset=dataset, feature_keys=["conditions", "procedures", "drugs_hist"],
                                     label_key="drugs", mode="multilabel")
        self.embedding_dim = embedding_dim
        self.drug_tokenizer = self.get_label_tokenizer()

        # 1. 基础支路
        self.cond_embeddings = nn.Embedding(self.get_feature_tokenizers()['conditions'].get_vocabulary_size(),
                                            embedding_dim, padding_idx=0)
        self.proc_embeddings = nn.Embedding(self.get_feature_tokenizers()['procedures'].get_vocabulary_size(),
                                            embedding_dim, padding_idx=0)
        self.drug_embeddings = nn.Embedding(self.drug_tokenizer.get_vocabulary_size(), embedding_dim, padding_idx=0)

        # 2. 外部投影 (增加 Dropout 防止过拟合)
        self.sem_proj = nn.Sequential(nn.Linear(768, 512), nn.Dropout(0.1))
        self.mol_proj = nn.Sequential(nn.Linear(768, 512), nn.Dropout(0.1))

        # 3. 三模态融合与自适应对齐
        self.hier_fusion = HierarchicalModalFusion(dim=512)
        self.adaptive_align = AdaptiveMultiHeadAlignment(dim=512)

        # 4. 后端优化
        # 减少输入维度到 embedding_dim * 2，只保留 [融合后特征, 原始 ID 特征]
        self.bi_gru = nn.GRU(embedding_dim * 2, embedding_dim, batch_first=True, bidirectional=True)
        self.label_attn = LabelAttention(embedding_dim * 2, embedding_dim, self.drug_tokenizer.get_vocabulary_size())

        # 知识库
        self.pre_cond = self.load_dict(r"D:\Code\LAMRec-RAGBK\models\MIMIC3_DIAGNOSES_Embeddings_512_ID.csv",
                                       "ICD9_CODE")
        self.pre_proc = self.load_dict(r"D:\Code\LAMRec-RAGBK\models\MIMIC3_PROCEDURES_Embeddings_512_ID.csv",
                                       "ICD9_CODE")
        self.pre_sem = self.load_dict(r"D:\Code\LAMRec-RAGBK\models\MIMIC3_Drug_Embeddings_768_ID.csv", "code")
        self.pre_mol = self.load_dict(r"D:\Code\LAMRec-RAGBK\models\aligned_drugs_and_smiles_features_768.csv", "ATC",
                                      is_mol=True)

        self.loss_fn = DiceBCELoss(from_logits=True, bce_weight=bce_weight)
        self.ddi_adj = torch.tensor(
            pd.read_csv(r"D:\Code\LAMRec-RAGBK\models\ddi_adj_final_131.csv", header=None).values, dtype=torch.float32)
        self.alpha = kwargs.get("alpha", 0.07)

    def load_dict(self, path, key_col, is_mol=False):
        df = pd.read_csv(path)
        return {str(row[key_col]): torch.tensor(
            ast.literal_eval(row["Embedding"]) if not is_mol else [float(x) for x in row["aggregated_vector"].split()],
            dtype=torch.float) for _, row in df.iterrows()}

    def get_dynamic_k_emb(self, batch_data, k_dict, dim, T):
        batch_res = []
        for patient in batch_data:
            visit_res = []
            for visit in patient:
                codes = [c for c in (visit if isinstance(visit, list) else [visit]) if c != "<pad>"]
                vecs = [k_dict.get(str(c), torch.zeros(dim)) for c in codes]
                visit_res.append(torch.stack(vecs).mean(0) if vecs else torch.zeros(dim))
            while len(visit_res) < T: visit_res.append(torch.zeros(dim))
            batch_res.append(torch.stack(visit_res[:T]))
        return torch.stack(batch_res).to(self.device)

    def forward(self, conditions, procedures, drugs_hist, drugs, **kwargs):
        c_ids = torch.tensor(self.get_feature_tokenizers()["conditions"].batch_encode_3d(conditions),
                             device=self.device)
        p_ids = torch.tensor(self.get_feature_tokenizers()["procedures"].batch_encode_3d(procedures),
                             device=self.device)
        T = c_ids.size(1)

        # 1. Temporal Branch (使用拼接而非加权和，避免信息丢失)
        x_t_c = masked_mean(self.cond_embeddings(c_ids), c_ids, 0)
        x_t_p = masked_mean(self.proc_embeddings(p_ids), p_ids, 0)
        # 时间序列的基础表示：[B, T, D]
        x_temporal = (x_t_c + x_t_p) / 2

        # 2. Knowledge Branches
        s_c = self.get_dynamic_k_emb(conditions, self.pre_cond, 512, T)
        s_p = self.get_dynamic_k_emb(procedures, self.pre_proc, 512, T)
        s_d = self.sem_proj(self.get_dynamic_k_emb(drugs_hist, self.pre_sem, 768, T))
        x_semantic = (s_c + s_p + s_d) / 3
        x_struct = self.mol_proj(self.get_dynamic_k_emb(drugs_hist, self.pre_mol, 768, T))

        # 3. Fusion & Alignment
        f_h = self.hier_fusion(x_temporal, x_semantic, x_struct)
        f_a = self.adaptive_align(f_h, x_semantic, x_semantic)
        combined = f_h + 0.1 * f_a  # 降低 Alignment 权重，以 Fusion 为主，避免过度对齐

        # 4. Predict (精简输入，只保留关键的融合特征和历史药物特征)
        # 获取平铺的历史药物 Embedding
        h_idx = [self.drug_tokenizer.convert_tokens_to_indices(
            [c for v in p for c in (v if isinstance(v, list) else [v]) if c != "<pad>"]) for p in drugs_hist]
        max_h = max(len(x) for x in h_idx) if h_idx else 1
        h_ids = torch.tensor([x + [0] * (max_h - len(x)) for x in h_idx], device=self.device)
        x_t_h = self.drug_embeddings(h_ids).sum(dim=1, keepdim=True).expand(-1, T, -1)

        seq_in = torch.cat([combined, x_t_h], dim=-1)  # 维度 512*2

        gru_out, _ = self.bi_gru(seq_in)
        logits = self.label_attn(gru_out)

        y_true = self.prepare_labels(drugs, self.drug_tokenizer)
        loss = self.loss_fn(logits, y_true)

        y_prob = torch.sigmoid(logits)
        if ddi_rate_score([np.where(s >= 0.5)[0] for s in y_prob.detach().cpu().numpy()],
                          self.ddi_adj.cpu().numpy()) >= 0.06:
            loss += self.alpha * (
                        torch.sum((y_prob.T @ y_prob).mul(self.ddi_adj.to(self.device))) / self.ddi_adj.shape[0] ** 2)

        return {"loss": loss, "y_prob": y_prob, "y_true": y_true}

# masked_mean 函数保持不变
def masked_mean(embed, ids, padding_idx):
    mask = (ids != padding_idx).float()
    return (embed * mask.unsqueeze(-1)).sum(dim=2) / mask.sum(dim=2, keepdim=True).clamp_min(1.0)