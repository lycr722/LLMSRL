import os
from pathlib import Path
from typing import List, Dict, Optional, Callable, Type
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pyhealth.medcode import ATC
from pyhealth.models import BaseModel
from pyhealth.datasets import SampleEHRDataset
from pyhealth.metrics import ddi_rate_score
from models.layers import TransformerCrossAttn, LabelAttention
from models.losses import DiceBCELoss
from collections import defaultdict
import random


class MLPMixer(nn.Module):

    def __init__(self, dim: int, num_tokens: int, expansion_factor: int = 2, dropout: float = 0.1):
        super().__init__()

        self.token_mixer = nn.Sequential(
            nn.LayerNorm(num_tokens),
            nn.Linear(num_tokens, num_tokens * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_tokens * expansion_factor, num_tokens),
            nn.Dropout(dropout)
        )

        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_transposed = x.transpose(1, 2)  # -> [B, D, N]
        mixed_tokens = self.token_mixer(x_transposed)
        x = x + mixed_tokens.transpose(1, 2)  # 残差连接

        # Channel mixing
        mixed_channels = self.channel_mixer(x)
        x = x + mixed_channels  # 残差连接

        return x


class AttentionGatedFusion(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_scores = self.attention_net(x).squeeze(-1)  # -> [B, N]
        attn_weights = torch.softmax(attn_scores, dim=1)  # -> [B, N]
        fused_emb = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)

        return fused_emb

class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttentionPooling, self).__init__()
        self.attn = nn.Linear(input_dim, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        attn_hidden = torch.tanh(self.attn(x))
        attn_scores = self.context_vector(attn_hidden).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)
        return pooled


class BiDirectionalCrossAttentionFusion(nn.Module):
    def __init__(self, orig_dim: int, enh_dim: int, fuse_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.orig_proj = nn.Sequential(nn.Linear(orig_dim, fuse_dim), nn.GELU(), nn.LayerNorm(fuse_dim))
        self.enh_proj = nn.Sequential(nn.Linear(enh_dim, fuse_dim), nn.GELU(), nn.LayerNorm(fuse_dim))
        self.attn_orig_to_enh = nn.MultiheadAttention(embed_dim=fuse_dim, num_heads=num_heads, dropout=dropout,
                                                      batch_first=True)
        self.attn_enh_to_orig = nn.MultiheadAttention(embed_dim=fuse_dim, num_heads=num_heads, dropout=dropout,
                                                      batch_first=True)
        self.gate_fn = nn.Linear(2 * fuse_dim, fuse_dim)
        self.layer_norm = nn.LayerNorm(fuse_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, orig_emb: torch.Tensor, enh_emb: torch.Tensor) -> torch.Tensor:
        x, y = self.orig_proj(orig_emb), self.enh_proj(enh_emb)
        attn_out1, _ = self.attn_orig_to_enh(query=x, key=y, value=y)
        attn_out2, _ = self.attn_enh_to_orig(query=y, key=x, value=x)

        def gated_fuse(residual: torch.Tensor, attn_output: torch.Tensor) -> torch.Tensor:
            combined = torch.cat([residual + attn_output, residual], dim=-1)
            g = torch.sigmoid(self.gate_fn(combined))
            return g * (residual + attn_output) + (1 - g) * residual

        fused_orig, fused_enh = gated_fuse(x, attn_out1), gated_fuse(y, attn_out2)
        return self.dropout(self.layer_norm(fused_orig + fused_enh))


class LAMRec(BaseModel):
    def __init__(self, dataset: SampleEHRDataset, embedding_dim: int, bce_weight: float = 0.5, **kwargs):
        super(LAMRec, self).__init__(
            dataset=dataset,
            feature_keys=["conditions", "procedures", "drugs_hist"],
            label_key="drugs",
            mode="multilabel",
        )
        self.feat_tokenizers = self.get_feature_tokenizers()
        self.label_tokenizer = self.get_label_tokenizer()
        self.drug_tokenizer = self.label_tokenizer
        self.dataset = dataset
        self.embedding_dim = embedding_dim

        self.alpha =0.07

        self.conditions_embeddings = nn.Embedding(self.feat_tokenizers['conditions'].get_vocabulary_size(),
                                                  embedding_dim,
                                                  padding_idx=self.feat_tokenizers['conditions'].get_padding_index())
        self.procedures_embeddings = nn.Embedding(self.feat_tokenizers['procedures'].get_vocabulary_size(),
                                                  embedding_dim,
                                                  padding_idx=self.feat_tokenizers['procedures'].get_padding_index())
        self.drugs_embeddings = nn.Embedding(self.drug_tokenizer.get_vocabulary_size(), embedding_dim, padding_idx=0)   #drugs_embedding=Embedding(131, 512, padding_idx=0)
        self.label_wise_attention = LabelAttention(embedding_dim * 2, embedding_dim,
                                                   self.drug_tokenizer.get_vocabulary_size())


        gru_hidden = kwargs.get("gru_hidden_size", embedding_dim)
        self.bi_gru = nn.GRU(input_size=embedding_dim * 3, hidden_size=gru_hidden,
                             num_layers=kwargs.get("gru_layers", 1), batch_first=True, bidirectional=True)
        self.task_loss_fn = DiceBCELoss(from_logits=True, bce_weight=bce_weight)
        self.ddi_adj = self.generate_ddi_adj().to(self.device)

        # 保存文件的逻辑保持不变
        BASE_CACHE_PATH = os.path.join(str(Path.home()), ".cache/pyhealth/")
        if not os.path.exists(BASE_CACHE_PATH):
            os.makedirs(BASE_CACHE_PATH)
        np.save(os.path.join(BASE_CACHE_PATH, "ddi_adj.npy"), self.ddi_adj.cpu().numpy())

        self.use_llm = kwargs.get("use_llm", True)
        if self.use_llm:
            print("Using LLM enhancement module.")
            self.semantic_dim = 768  # 假设您的语义文件也是768维
            self.molecular_dim = 768  # 这是您新分子向量的维度
            self.target_dim = kwargs.get("llm_target_dim", 512)
            self.max_codes_per_visit_condition = kwargs.get("max_codes_per_visit_condition", 24)
            self.max_codes_per_visit_procedures = kwargs.get("max_codes_per_visit_procedures", 12)

            self.precomputed_conditions = self.load_precomputed_embeddings(
                r"D:\Down\LAMRec-RAG\models\DIAGNOSES_Official_Embedding_512_large.csv", "ICD9_CODE")
            self.precomputed_procedures = self.load_precomputed_embeddings(
                r"D:\Down\LAMRec-RAG\models\PROCEDURES_Official_Embedding_512_large.csv", "ICD9_CODE")
            print("正在加载药物语义嵌入...")
            self.precomputed_drugs_semantic = self.load_semantic_embeddings(
                r"D:\Down\LAMRec-RAG\models\filtered_drug_embeddings_768.csv", # <-- 语义文件路径
                target_dim=self.semantic_dim
            )
            print("正在加载药物分子嵌入...")
            self.precomputed_drugs_molecular = self.load_molecular_embeddings(
                r"D:\Down\LAMRec-RAG\models\aligned_drugs_and_smiles_features_768.csv", # <-- 分子文件路径
                target_dim=self.molecular_dim
            )

            self.attn_pool_conditions = SelfAttentionPooling(self.target_dim, self.target_dim // 2)
            self.attn_pool_procedures = SelfAttentionPooling(self.target_dim, self.target_dim // 2)
            self.norm_orig_cond = nn.LayerNorm(embedding_dim);
            self.norm_orig_proc = nn.LayerNorm(embedding_dim);
            self.norm_orig_drug = nn.LayerNorm(embedding_dim)
            self.norm_enh_cond = nn.LayerNorm(self.target_dim);
            self.norm_enh_proc = nn.LayerNorm(self.target_dim);
            num_heads = kwargs.get('heads', 1)
            self.cross_attn_fusion_conditions = BiDirectionalCrossAttentionFusion(embedding_dim, self.target_dim,
                                                                                  embedding_dim, num_heads)
            self.cross_attn_fusion_procedures = BiDirectionalCrossAttentionFusion(embedding_dim, self.target_dim,
                                                                                  embedding_dim, num_heads)

            # 语义模块
            self.attn_pool_semantic = SelfAttentionPooling(self.semantic_dim, self.semantic_dim // 2)
            self.norm_enh_semantic = nn.LayerNorm(self.semantic_dim)
            self.cross_attn_fusion_semantic = BiDirectionalCrossAttentionFusion(
                embedding_dim, self.semantic_dim, embedding_dim, num_heads
            )

            # 分子模块
            self.attn_pool_molecular = SelfAttentionPooling(self.molecular_dim, self.molecular_dim // 2)
            self.norm_enh_molecular = nn.LayerNorm(self.molecular_dim)
            self.cross_attn_fusion_molecular = BiDirectionalCrossAttentionFusion(
                embedding_dim, self.molecular_dim, embedding_dim, num_heads
            )

            print("Initializing Multimodal Fusion Modules (MLP-Mixer & Attention Gated Fusion)...")
            self.drug_fusion_mixer = MLPMixer(
                dim=embedding_dim,
                num_tokens=2  # 2个模态: 语义和分子
            )
            self.drug_fusion_gate = AttentionGatedFusion(
                dim=embedding_dim
            )



    def load_precomputed_embeddings(self, filepath: str, code_key: str) -> Dict[str, torch.Tensor]:
        df = pd.read_csv(filepath, dtype={code_key: str})
        emb_dict = {}
        for _, row in df.iterrows():
            code, vec_str = row[code_key], row["Embedding"]
            vec = torch.tensor(ast.literal_eval(vec_str), dtype=torch.float)
            if vec.size(0) != self.target_dim:
                if vec.size(0) < self.target_dim:
                    vec = F.pad(vec, (0, self.target_dim - vec.size(0)))
                else:
                    vec = vec[:self.target_dim]
            emb_dict[code] = nn.Parameter(vec, requires_grad=True)
        return nn.ParameterDict(emb_dict)

    def parse_molecular_vector(self, vector_str: str) -> Optional[np.ndarray]:
        """用于解析以空格分隔的向量字符串的辅助函数。"""
        try:
            return np.array([float(x) for x in vector_str.split()])
        except (ValueError, AttributeError):
            return None


    def load_semantic_embeddings(self, filepath: str, target_dim: int) -> nn.ParameterDict:
        df = pd.read_csv(filepath)
        emb_dict = {}
        for _, row in df.iterrows():
            code, vec_str = row["code"], row["Embedding"]
            vec = torch.tensor(ast.literal_eval(vec_str), dtype=torch.float)
            if vec.size(0) == target_dim:
                emb_dict[code] = nn.Parameter(vec, requires_grad=True)
        return nn.ParameterDict(emb_dict)
    def load_molecular_embeddings(self, filepath: str, target_dim: int) -> nn.ParameterDict:
        """从指定的CSV文件加载预计算的分子嵌入。"""
        df = pd.read_csv(filepath)
        emb_dict = {}
        for _, row in df.iterrows():
            # 使用您新文件中的列名："ATC" 和 "aggregated_vector"
            code, vec_str = row["ATC"], row["aggregated_vector"]
            vec_np = self.parse_molecular_vector(vec_str)
            # 确保向量被正确解析且维度符合预期
            if vec_np is not None and len(vec_np) == target_dim:
                emb_dict[code] = nn.Parameter(torch.tensor(vec_np, dtype=torch.float), requires_grad=True)
        return nn.ParameterDict(emb_dict)
    def generate_ddi_adj(self) -> torch.tensor:
        file_path = r"D:\Down\LAMRec-RAG\models\ddi_adj_final_131.csv"
        ddi_adj_df = pd.read_csv(file_path, header=None)
        ddi_adj = torch.tensor(ddi_adj_df.values, dtype=torch.float32)
        return ddi_adj

    def forward(self, conditions: List[List[List[str]]], procedures: List[List[List[str]]], drugs: List[List[str]],
                drugs_hist: List[List[str]],
                **kwargs) -> Dict[str, torch.Tensor]:

        cond_ids = torch.tensor(self.feat_tokenizers["conditions"].batch_encode_3d(conditions),
                                dtype=torch.long, device=self.device)
        proc_ids = torch.tensor(self.feat_tokenizers["procedures"].batch_encode_3d(procedures),
                                dtype=torch.long, device=self.device)
        conditions_emb = masked_mean(self.conditions_embeddings(cond_ids), cond_ids,
                                     self.feat_tokenizers['conditions'].get_padding_index())
        procedures_emb = masked_mean(self.procedures_embeddings(proc_ids), proc_ids,
                                     self.feat_tokenizers['procedures'].get_padding_index())

        drugs_indices_jagged = []
        for patient_codes in drugs_hist:
            flat_codes = []
            for item in patient_codes:
                if item == "<pad>": continue
                if isinstance(item, list):
                    flat_codes.extend(item)
                else:
                    flat_codes.append(item)
            drugs_indices_jagged.append(self.drug_tokenizer.convert_tokens_to_indices(flat_codes))

        max_drug_len = max(len(x) for x in drugs_indices_jagged) if drugs_indices_jagged else 0
        padded_drug_indices = [x + [0] * (max_drug_len - len(x)) for x in drugs_indices_jagged]
        drugs_ids = torch.tensor(padded_drug_indices, dtype=torch.long, device=self.device)

        drugs_emb = torch.sum(self.drugs_embeddings(drugs_ids), dim=1)

        # --- LLM Fusion for conditions and procedures ---
        if self.use_llm:
            cond_text_emb = self.get_llm_embedding(conditions, self.precomputed_conditions, self.attn_pool_conditions,
                                                   self.max_codes_per_visit_condition, conditions_emb.shape[1])
            proc_text_emb = self.get_llm_embedding(procedures, self.precomputed_procedures, self.attn_pool_procedures,
                                                   self.max_codes_per_visit_procedures, procedures_emb.shape[1])
            conditions_emb = self.cross_attn_fusion_conditions(self.norm_orig_cond(conditions_emb),
                                                               self.norm_enh_cond(cond_text_emb))
            procedures_emb = self.cross_attn_fusion_procedures(self.norm_orig_proc(procedures_emb),
                                                               self.norm_enh_proc(proc_text_emb))

            orig_drug_emb_normalized = self.norm_orig_drug(drugs_emb).unsqueeze(1)

            # --- 通路 A: 语义专家 (Semantic Expert Pathway) ---
            sem_code_embs = self.get_external_embedding(drugs_hist, self.precomputed_drugs_semantic, self.semantic_dim)
            sem_enh_emb = self.attn_pool_semantic(sem_code_embs)
            semantic_fused_emb = self.cross_attn_fusion_semantic(
                orig_drug_emb_normalized,
                self.norm_enh_semantic(sem_enh_emb).unsqueeze(1)
            ).squeeze(1)

            # --- 消融实验: 仅使用语义增强 ---
            # 分子通路和多模态融合模块已被移除。
            drugs_emb = semantic_fused_emb

        T = conditions_emb.size(1)
        drugs_emb_expanded = drugs_emb.unsqueeze(1).expand(-1, T, -1)
        seq_input = torch.cat([conditions_emb, procedures_emb, drugs_emb_expanded], dim=-1)

        gru_out, _ = self.bi_gru(seq_input)
        logits = self.label_wise_attention(gru_out)
        curr_drugs_labels = self.prepare_labels(drugs, self.label_tokenizer)
        loss = self.task_loss_fn(logits, curr_drugs_labels)
        y_prob = torch.sigmoid(logits)
        y_pred = y_prob.detach().cpu().numpy()
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        y_pred_indices = [list(np.where(sample == 1)[0]) for sample in y_pred]
        y_true_indices = [list(np.where(sample.cpu().numpy() == 1)[0]) for sample in curr_drugs_labels]

        y_pred = [np.where(sample == 1)[0] for sample in y_pred]
        current_ddi_rate = ddi_rate_score(y_pred, self.ddi_adj.cpu().numpy())

        if current_ddi_rate >= 0.06:
            mul_pred_prob = y_prob.T @ y_prob  # (voc_size, voc_size)
            batch_ddi_loss = (
                    torch.sum(mul_pred_prob.mul(self.ddi_adj.to(self.device))) / self.ddi_adj.shape[0] ** 2
            )
            loss += self.alpha * batch_ddi_loss

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": curr_drugs_labels,
            "y_pred_indices": y_pred_indices,
            "y_true_indices": y_true_indices,
        }

    def get_llm_embedding(self, patients_data: List[List[List[str]]], precomputed_embeddings: nn.ParameterDict,
                          attn_pool: nn.Module, max_codes: int, T_target: int) -> torch.Tensor:
        batch_emb_list = []
        for patient_visits in patients_data:
            visit_vecs = []
            for visit_codes in patient_visits:
                code_vecs = [precomputed_embeddings.get(c, torch.zeros(self.target_dim, device=self.device)) for c in
                             visit_codes if c != "<pad>"]
                if not code_vecs: code_vecs = [torch.zeros(self.target_dim, device=self.device)]

                if len(code_vecs) < max_codes:
                    code_vecs.extend([torch.zeros(self.target_dim, device=self.device)] * (max_codes - len(code_vecs)))
                else:
                    code_vecs = code_vecs[:max_codes]

                visit_vecs.append(attn_pool(torch.stack(code_vecs, dim=0).unsqueeze(0)).squeeze(0))

            if len(visit_vecs) < T_target:
                visit_vecs.extend([torch.zeros(self.target_dim, device=self.device)] * (T_target - len(visit_vecs)))
            else:
                visit_vecs = visit_vecs[:T_target]

            batch_emb_list.append(torch.stack(visit_vecs, dim=0))
        return torch.stack(batch_emb_list, dim=0)


    def get_external_embedding(self,
                               patients_data: List[List[str]],
                               precomputed_dict: nn.ParameterDict,
                               target_dim: int) -> torch.Tensor:
        """
        通用的外部嵌入获取函数。

        Args:
            patients_data: 批次的病人用药历史, e.g., drugs_hist.
            precomputed_dict: 用于查找的预计算嵌入字典 (可以是语义或分子).
            target_dim: 对应嵌入的维度 (例如 512, 768).

        Returns:
            一个形状为 [B, max_codes, D_enh] 的张量, 其中 B 是批次大小,
            max_codes 是批次中病人拥有的最大独立药品数, D_enh 是 target_dim.
        """
        batch_emb_list = []
        codes_no_padding = []

        # 步骤 1: 为每个病人提取并去重药品编码
        for patient_codes in patients_data:
            flat_codes = [
                code for item in patient_codes
                for code in (item if isinstance(item, list) else [item])
                if code != "<pad>"
            ]
            codes_no_padding.append(list(set(flat_codes)))

        # 步骤 2: 找到批次中单个病人的最大药品数，用于后续的填充
        max_codes = max(len(codes) for codes in codes_no_padding) if codes_no_padding and any(codes_no_padding) else 0

        # 步骤 3: 为每个病人查找向量并进行填充
        for codes in codes_no_padding:
            if not codes:
                # 处理没有历史用药的病人, 创建一个占位符向量
                code_vecs = [torch.zeros(target_dim, device=self.device)]
            else:
                # 从传入的字典中查找向量，如果找不到则使用零向量
                code_vecs = [
                    precomputed_dict.get(c, torch.zeros(target_dim, device=self.device))
                    for c in codes
                ]

            # 对每个病人的code数量进行填充，以保证张量形状统一
            if len(code_vecs) < max_codes:
                code_vecs.extend([torch.zeros(target_dim, device=self.device)] * (max_codes - len(code_vecs)))
            # 如果药品数超过最大值则截断 (虽然基于当前逻辑不会发生, 但这是个好习惯)
            else:
                code_vecs = code_vecs[:max_codes]

            batch_emb_list.append(torch.stack(code_vecs, dim=0))

        # 步骤 4: 处理整个批次都没有任何历史用药的极端情况
        if not batch_emb_list:
            print("整个批次都没有任何历史用药的极端情况")
            return torch.zeros((len(patients_data), 0, target_dim), device=self.device)

        return torch.stack(batch_emb_list, dim=0)

def masked_mean(embed, ids, padding_idx):

    # embed: [B, T, L, D] from embedding(ids)
    mask = (ids != padding_idx).float()  # [B,T,L]
    denom = mask.sum(dim=2, keepdim=True).clamp_min(1.0)
    summed = (embed * mask.unsqueeze(-1)).sum(dim=2)
    return summed / denom  # [B,T,D]0