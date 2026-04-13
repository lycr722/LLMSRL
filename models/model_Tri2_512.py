import os
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pyhealth.models import BaseModel
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
        # 第一阶段：时间序列 + 语义
        self.cam1 = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gate1 = nn.Sequential(nn.Linear(dim * 2, 1), nn.Sigmoid())

        # 第二阶段：中间特征 + 分子结构
        self.cam2 = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gate2 = nn.Sequential(nn.Linear(dim * 2, 1), nn.Sigmoid())

        self.norm = nn.LayerNorm(dim)

    def forward(self, x_temp, x_sem, x_struct):
        # 语义对齐与残差融合
        attn1, _ = self.cam1(x_temp, x_sem, x_sem)
        g1 = self.gate1(torch.cat([x_temp, attn1], dim=-1))
        f1 = self.norm(x_temp + g1 * attn1)

        # 结构对齐与残差融合
        attn2, _ = self.cam2(f1, x_struct, x_struct)
        g2 = self.gate2(torch.cat([f1, attn2], dim=-1))
        return self.norm(f1 + g2 * attn2)


class LAMRec(BaseModel):
    def __init__(self, dataset, embedding_dim=512, bce_weight=0.5, **kwargs):
        super(LAMRec, self).__init__(dataset=dataset, feature_keys=["conditions", "procedures", "drugs_hist"],
                                     label_key="drugs", mode="multilabel")
        self.embedding_dim = embedding_dim
        self.drug_tokenizer = self.get_label_tokenizer()

        # 1. 基础 Embedding
        self.cond_embeddings = nn.Embedding(self.get_feature_tokenizers()['conditions'].get_vocabulary_size(),
                                            embedding_dim, padding_idx=0)
        self.proc_embeddings = nn.Embedding(self.get_feature_tokenizers()['procedures'].get_vocabulary_size(),
                                            embedding_dim, padding_idx=0)
        self.drug_embeddings = nn.Embedding(self.drug_tokenizer.get_vocabulary_size(), embedding_dim, padding_idx=0)

        # 2. 知识投影层 (此时输入维度已改为 512)
        self.sem_proj = nn.Linear(512, 512)  # 对齐 512 维语义
        self.mol_proj = nn.Linear(768, 512)  # 分子暂维持 768

        # 3. 核心融合
        self.hier_fusion = HierarchicalModalFusion(dim=512)

        # 4. 动态混合门控 (初始设定为倾向原始 Baseline)
        self.alpha_gate = nn.Parameter(torch.tensor([-1.0]))

        # 5. 后端
        self.bi_gru = nn.GRU(embedding_dim * 3, embedding_dim, batch_first=True, bidirectional=True)
        self.label_attn = LabelAttention(embedding_dim * 2, embedding_dim, self.drug_tokenizer.get_vocabulary_size())

        # 6. 加载 512 维知识字典
        self.pre_cond = self.load_dict(r"D:\Code\LAMRec-RAGBK\models\MIMIC3_DIAGNOSES_Embeddings_512_ID.csv",
                                       "ICD9_CODE")
        self.pre_proc = self.load_dict(r"D:\Code\LAMRec-RAGBK\models\MIMIC3_PROCEDURES_Embeddings_512_ID.csv",
                                       "ICD9_CODE")
        self.pre_sem = self.load_dict(r"D:\Code\LAMRec-RAGBK\models\MIMIC3_Drug_Embeddings_512_ID.csv",
                                      "code")  # 更新为 512 路径
        self.pre_mol = self.load_dict(r"D:\Code\LAMRec-RAGBK\models\aligned_drugs_and_smiles_features_768.csv", "ATC",
                                      is_mol=True)

        self.loss_fn = DiceBCELoss(from_logits=True, bce_weight=bce_weight)
        self.ddi_adj = torch.tensor(
            pd.read_csv(r"D:\Code\LAMRec-RAGBK\models\ddi_adj_final_131.csv", header=None).values, dtype=torch.float32)
        self.alpha_ddi = kwargs.get("alpha", 0.07)

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

        # 1. Baseline 原始特征
        x_t_c = masked_mean(self.cond_embeddings(c_ids), c_ids, 0)
        x_t_p = masked_mean(self.proc_embeddings(p_ids), p_ids, 0)
        x_base = (x_t_c + x_t_p) / 2

        # 2. 三模态支路构建 (语义支路已设为 512 维输入)
        s_c = self.get_dynamic_k_emb(conditions, self.pre_cond, 512, T)
        s_p = self.get_dynamic_k_emb(procedures, self.pre_proc, 512, T)
        s_d = self.sem_proj(self.get_dynamic_k_emb(drugs_hist, self.pre_sem, 512, T))  # 512维对齐
        x_semantic = (s_c + s_p + s_d) / 3
        x_struct = self.mol_proj(self.get_dynamic_k_emb(drugs_hist, self.pre_mol, 768, T))

        # 3. 融合补强
        fused = self.hier_fusion(x_base, x_semantic, x_struct)

        # 4. 全局混合
        alpha = torch.sigmoid(self.alpha_gate)
        combined = (1 - alpha) * x_base + alpha * fused

        # 5. 历史 ID 支路 (平铺处理)
        h_idx = [self.drug_tokenizer.convert_tokens_to_indices(
            [c for v in p for c in (v if isinstance(v, list) else [v]) if c != "<pad>"]) for p in drugs_hist]
        max_h = max(len(x) for x in h_idx) if h_idx else 1
        h_ids = torch.tensor([x + [0] * (max_h - len(x)) for x in h_idx], device=self.device)
        x_t_h = self.drug_embeddings(h_ids).sum(dim=1, keepdim=True).expand(-1, T, -1)

        # 拼接进入 GRU
        seq_in = torch.cat([combined, x_base, x_t_h], dim=-1)
        gru_out, _ = self.bi_gru(seq_in)
        logits = self.label_attn(gru_out)

        y_true = self.prepare_labels(drugs, self.drug_tokenizer)
        loss = self.loss_fn(logits, y_true)

        y_prob = torch.sigmoid(logits)
        if ddi_rate_score([np.where(s >= 0.5)[0] for s in y_prob.detach().cpu().numpy()],
                          self.ddi_adj.cpu().numpy()) >= 0.06:
            loss += self.alpha_ddi * (
                        torch.sum((y_prob.T @ y_prob).mul(self.ddi_adj.to(self.device))) / self.ddi_adj.shape[0] ** 2)

        return {"loss": loss, "y_prob": y_prob, "y_true": y_true}


def masked_mean(embed, ids, padding_idx):
    mask = (ids != padding_idx).float()
    return (embed * mask.unsqueeze(-1)).sum(dim=2) / mask.sum(dim=2, keepdim=True).clamp_min(1.0)