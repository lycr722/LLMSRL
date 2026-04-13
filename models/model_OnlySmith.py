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
        x_transposed = x.transpose(1, 2)
        mixed_tokens = self.token_mixer(x_transposed)
        x = x + mixed_tokens.transpose(1, 2)

        mixed_channels = self.channel_mixer(x)
        x = x + mixed_channels

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
        attn_scores = self.attention_net(x).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
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
        self.alpha = 0.07

        self.conditions_embeddings = nn.Embedding(self.feat_tokenizers['conditions'].get_vocabulary_size(),
                                                  embedding_dim,
                                                  padding_idx=self.feat_tokenizers['conditions'].get_padding_index())
        self.procedures_embeddings = nn.Embedding(self.feat_tokenizers['procedures'].get_vocabulary_size(),
                                                  embedding_dim,
                                                  padding_idx=self.feat_tokenizers['procedures'].get_padding_index())
        self.drugs_embeddings = nn.Embedding(self.drug_tokenizer.get_vocabulary_size(), embedding_dim, padding_idx=0)
        self.label_wise_attention = LabelAttention(embedding_dim * 2, embedding_dim,
                                                   self.drug_tokenizer.get_vocabulary_size())

        gru_hidden = kwargs.get("gru_hidden_size", embedding_dim)
        self.bi_gru = nn.GRU(input_size=embedding_dim * 3, hidden_size=gru_hidden,
                             num_layers=kwargs.get("gru_layers", 1), batch_first=True, bidirectional=True)
        self.task_loss_fn = DiceBCELoss(from_logits=True, bce_weight=bce_weight)
        self.ddi_adj = self.generate_ddi_adj().to(self.device)

        BASE_CACHE_PATH = os.path.join(str(Path.home()), ".cache/pyhealth/")
        if not os.path.exists(BASE_CACHE_PATH):
            os.makedirs(BASE_CACHE_PATH)
        np.save(os.path.join(BASE_CACHE_PATH, "ddi_adj.npy"), self.ddi_adj.cpu().numpy())

        self.use_llm = kwargs.get("use_llm", True)
        if self.use_llm:
            print("Using LLM enhancement module (Molecular-Only).")
            self.molecular_dim = 768
            self.target_dim = kwargs.get("llm_target_dim", 512)
            self.max_codes_per_visit_condition = kwargs.get("max_codes_per_visit_condition", 24)
            self.max_codes_per_visit_procedures = kwargs.get("max_codes_per_visit_procedures", 12)

            self.precomputed_conditions = self.load_precomputed_embeddings(
                r"D:\Down\LAMRec-RAG\models\DIAGNOSES_Official_Embedding_512_large.csv", "ICD9_CODE")
            self.precomputed_procedures = self.load_precomputed_embeddings(
                r"D:\Down\LAMRec-RAG\models\PROCEDURES_Official_Embedding_512_large.csv", "ICD9_CODE")

            # --- MODIFICATION START ---
            # Removed loading of semantic embeddings
            print("正在加载药物分子嵌入...")
            self.precomputed_drugs_molecular = self.load_molecular_embeddings(
                r"D:\Down\LAMRec-RAG\models\aligned_drugs_and_smiles_features_768.csv", target_dim=self.molecular_dim)
            # --- MODIFICATION END ---

            self.attn_pool_conditions = SelfAttentionPooling(self.target_dim, self.target_dim // 2)
            self.attn_pool_procedures = SelfAttentionPooling(self.target_dim, self.target_dim // 2)
            self.norm_orig_cond = nn.LayerNorm(embedding_dim)
            self.norm_orig_proc = nn.LayerNorm(embedding_dim)
            self.norm_orig_drug = nn.LayerNorm(embedding_dim)
            self.norm_enh_cond = nn.LayerNorm(self.target_dim)
            self.norm_enh_proc = nn.LayerNorm(self.target_dim)
            num_heads = kwargs.get('heads', 1)
            self.cross_attn_fusion_conditions = BiDirectionalCrossAttentionFusion(embedding_dim, self.target_dim,
                                                                                  embedding_dim, num_heads)
            self.cross_attn_fusion_procedures = BiDirectionalCrossAttentionFusion(embedding_dim, self.target_dim,
                                                                                  embedding_dim, num_heads)

            # --- MODIFICATION START ---
            # Removed all semantic-related modules
            # ---
            # --- Molecular module (remains)
            self.attn_pool_molecular = SelfAttentionPooling(self.molecular_dim, self.molecular_dim // 2)
            self.norm_enh_molecular = nn.LayerNorm(self.molecular_dim)
            self.cross_attn_fusion_molecular = BiDirectionalCrossAttentionFusion(embedding_dim, self.molecular_dim,
                                                                                 embedding_dim, num_heads)

            # Removed fusion modules (MLPMixer, AttentionGatedFusion)
            # --- MODIFICATION END ---

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
        try:
            return np.array([float(x) for x in vector_str.split()])
        except (ValueError, AttributeError):
            return None

    # This function is no longer needed but kept for completeness in case you switch back
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
        df = pd.read_csv(filepath)
        emb_dict = {}
        for _, row in df.iterrows():
            code, vec_str = row["ATC"], row["aggregated_vector"]
            vec_np = self.parse_molecular_vector(vec_str)
            if vec_np is not None and len(vec_np) == target_dim:
                emb_dict[code] = nn.Parameter(torch.tensor(vec_np, dtype=torch.float), requires_grad=True)
        return nn.ParameterDict(emb_dict)

    def generate_ddi_adj(self) -> torch.tensor:
        file_path = r"D:\Down\LAMRec-RAG\models\ddi_adj_final_131.csv"
        ddi_adj_df = pd.read_csv(file_path, header=None)
        ddi_adj = torch.tensor(ddi_adj_df.values, dtype=torch.float32)
        return ddi_adj

    def forward(self, conditions: List[List[List[str]]], procedures: List[List[List[str]]], drugs: List[List[str]],
                drugs_hist: List[List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        cond_ids = torch.tensor(self.feat_tokenizers["conditions"].batch_encode_3d(conditions), dtype=torch.long,
                                device=self.device)
        proc_ids = torch.tensor(self.feat_tokenizers["procedures"].batch_encode_3d(procedures), dtype=torch.long,
                                device=self.device)
        conditions_emb = masked_mean(self.conditions_embeddings(cond_ids), cond_ids,
                                     self.feat_tokenizers['conditions'].get_padding_index())
        procedures_emb = masked_mean(self.procedures_embeddings(proc_ids), proc_ids,
                                     self.feat_tokenizers['procedures'].get_padding_index())

        drugs_indices_jagged = []
        for patient_codes in drugs_hist:
            flat_codes = [item for sublist in patient_codes for item in
                          (sublist if isinstance(sublist, list) else [sublist]) if item != "<pad>"]
            drugs_indices_jagged.append(self.drug_tokenizer.convert_tokens_to_indices(flat_codes))

        max_drug_len = max(len(x) for x in drugs_indices_jagged) if drugs_indices_jagged else 0
        padded_drug_indices = [x + [0] * (max_drug_len - len(x)) for x in drugs_indices_jagged]
        drugs_ids = torch.tensor(padded_drug_indices, dtype=torch.long, device=self.device)
        drugs_emb_base = torch.sum(self.drugs_embeddings(drugs_ids), dim=1)

        final_drugs_emb = drugs_emb_base
        if self.use_llm:
            cond_text_emb = self.get_llm_embedding(conditions, self.precomputed_conditions, self.attn_pool_conditions,
                                                   self.max_codes_per_visit_condition, conditions_emb.shape[1])
            proc_text_emb = self.get_llm_embedding(procedures, self.precomputed_procedures, self.attn_pool_procedures,
                                                   self.max_codes_per_visit_procedures, procedures_emb.shape[1])
            conditions_emb = self.cross_attn_fusion_conditions(self.norm_orig_cond(conditions_emb),
                                                               self.norm_enh_cond(cond_text_emb))
            procedures_emb = self.cross_attn_fusion_procedures(self.norm_orig_proc(procedures_emb),
                                                               self.norm_enh_proc(proc_text_emb))

            orig_drug_emb_normalized = self.norm_orig_drug(drugs_emb_base).unsqueeze(1)

            # --- MODIFICATION START ---
            # Removed Path A (Semantic Expert)
            # ---
            # --- Path B (Molecular Structure Expert) - This is now the main path
            mol_code_embs = self.get_external_embedding(drugs_hist, self.precomputed_drugs_molecular,
                                                        self.molecular_dim)
            mol_enh_emb = self.attn_pool_molecular(mol_code_embs)
            molecular_fused_emb = self.cross_attn_fusion_molecular(
                orig_drug_emb_normalized,
                self.norm_enh_molecular(mol_enh_emb).unsqueeze(1)
            ).squeeze(1)

            # The final drug embedding is now directly the molecular-fused embedding
            final_drugs_emb = molecular_fused_emb
            # --- MODIFICATION END ---

        T = conditions_emb.size(1)
        drugs_emb_expanded = final_drugs_emb.unsqueeze(1).expand(-1, T, -1)
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

        y_pred_for_ddi = [np.where(sample == 1)[0] for sample in y_pred]
        current_ddi_rate = ddi_rate_score(y_pred_for_ddi, self.ddi_adj.cpu().numpy())

        if current_ddi_rate >= 0.06:
            mul_pred_prob = y_prob.T @ y_prob
            batch_ddi_loss = (torch.sum(mul_pred_prob.mul(self.ddi_adj.to(self.device))) / self.ddi_adj.shape[0] ** 2)
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

    def get_external_embedding(self, patients_data: List[List[str]], precomputed_dict: nn.ParameterDict,
                               target_dim: int) -> torch.Tensor:
        batch_emb_list = []
        codes_no_padding = []
        for patient_codes in patients_data:
            flat_codes = [code for item in patient_codes for code in (item if isinstance(item, list) else [item]) if
                          code != "<pad>"]
            codes_no_padding.append(list(set(flat_codes)))

        max_codes = max(len(codes) for codes in codes_no_padding) if codes_no_padding and any(codes_no_padding) else 0

        for codes in codes_no_padding:
            if not codes:
                code_vecs = [torch.zeros(target_dim, device=self.device)]
            else:
                code_vecs = [precomputed_dict.get(c, torch.zeros(target_dim, device=self.device)) for c in codes]

            if len(code_vecs) < max_codes:
                code_vecs.extend([torch.zeros(target_dim, device=self.device)] * (max_codes - len(code_vecs)))
            else:
                code_vecs = code_vecs[:max_codes]
            batch_emb_list.append(torch.stack(code_vecs, dim=0))

        if not batch_emb_list:
            print("Warning: Entire batch has no historical drug data.")
            return torch.zeros((len(patients_data), 0, target_dim), device=self.device)

        return torch.stack(batch_emb_list, dim=0)


def masked_mean(embed, ids, padding_idx):
    mask = (ids != padding_idx).float().unsqueeze(-1)
    denom = mask.sum(dim=2).clamp_min(1.0)
    summed = (embed * mask).sum(dim=2)
    return summed / denom