import os
import time
import numpy as np
from sentence_transformers import SentenceTransformer, models

# ==================== 配置参数 ====================
# 必须指向包含 config.json 和 pytorch_model.bin 的本地绝对路径
LOCAL_MODEL_PATH = "/pubmedbert-base-embeddings"  
EMBEDDING_DIM = 768

# ==================== PubMedBERT 初始化 ====================
print("=" * 80)
print("🔄 Loading PubMedBERT model from local path...")
print(f"   Path: {LOCAL_MODEL_PATH}")

try:
    # 1. 显式加载基础的 Transformer 模型
    word_embedding_model = models.Transformer(LOCAL_MODEL_PATH)
    
    # 2. 显式添加 Pooling 层（提取整段文本的平均特征向量）
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )
    
    # 3. 手动组合为 SentenceTransformer 模型
    pubmedbert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    print(f"✅ PubMedBERT model loaded successfully!")
    print(f"   Model dimension: {EMBEDDING_DIM}")
except Exception as e:
    print(f"❌ Failed to load model from {LOCAL_MODEL_PATH}: {e}")
    raise

# ==================== 核心函数 ====================

def get_drug_embedding(text: str) -> np.ndarray:
    """使用 PubMedBERT 将文本转化为向量"""
    try:
        # convert_to_numpy=True 保证输出标准的 numpy 数组
        embedding = pubmedbert_model.encode(text, convert_to_numpy=True)
        return embedding
    except Exception as e:
        print(f"❌ Error embedding text: {e}")
        return np.zeros(EMBEDDING_DIM)

# ==================== 测试流程 ====================

def test_text_to_vector():
    """测试将指定医学段落转换为向量"""
    
    # 你提供的测试文本
    sample_text = (
        "Proton pump inhibitors (ATC-3: A02A), a class of medications that inhibit gastric acid secretion, "
        "are primarily utilized in the management of gastroesophageal reflux disease (ICD-9-CM: 530.81) and "
        "peptic ulcer disease (ICD-9-CM: 531.0). These agents function by irreversibly binding to the H+/K+ "
        "ATPase enzyme in the gastric parietal cells, effectively blocking the final step of acid production, "
        "which leads to increased gastric pH and promotes healing of the gastric and esophageal mucosa. "
        "In clinical practice, proton pump inhibitors are often prescribed following procedures such as "
        "esophagogastroduodenoscopy (ICD-9-CM Proc: 45.16) to prevent stress-related mucosal disease. "
        "To enhance therapeutic outcomes, they are frequently co-prescribed with H2-receptor antagonists (ATC-3: A02BA), "
        "which provide an additive effect in reducing gastric acidity, particularly in patients with refractory symptoms. "
        "However, proton pump inhibitors are contraindicated in individuals with known hypersensitivity to the drug (ICD-9-CM: 995.3) "
        "and should be avoided in patients with severe liver impairment (ICD-9-CM: 571.5). A significant drug-drug "
        "interaction exists with clopidogrel (ATC-3: B01AC), an antiplatelet agent, as proton pump inhibitors can "
        "reduce the activation of clopidogrel, potentially diminishing its efficacy and increasing the risk of "
        "cardiovascular events; this necessitates careful consideration when managing patients requiring both therapies."
    )

    print("\n" + "=" * 80)
    print("🧪 Running Text-to-Vector Test")
    print("=" * 80)
    
    print(f"\n🔹 Input Text ({len(sample_text)} characters):")
    # 打印前200个字符作为预览
    print(f"   {sample_text[:200]}...\n")

    print("🔹 Generating embedding...")
    start_time = time.time()
    
    # 调用模型
    embedding = get_drug_embedding(sample_text)
    
    elapsed_time = time.time() - start_time
    print(f"⏱️ Generation time: {elapsed_time:.4f} seconds")
    
    print("\n🔹 Validating results...")
    print(f"📏 Expected Shape: ({EMBEDDING_DIM},)")
    print(f"📐 Actual Shape:   {embedding.shape}")
    
    if embedding.shape[0] == EMBEDDING_DIM:
        print("✅ Success! Dimension matches perfectly.")
    else:
        print("❌ Warning: Dimension mismatch.")
        
    print(f"\n🔢 Embedding Vector Preview (first 10 out of 768 dimensions):")
    # 打印前10个数值，格式化保留5位小数
    preview_values = ", ".join([f"{val:.5f}" for val in embedding[:10]])
    print(f"   [{preview_values}, ...]")
    
    print("\n" + "=" * 80)
    print("🏁 Test Completed Successfully!")
    print("=" * 80)

if __name__ == "__main__":
    test_text_to_vector()