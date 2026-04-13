import requests
import numpy as np
import itertools
from rouge_score import rouge_scorer

# =========================
# 1. API 配置 (复用您的配置)
# =========================
API_URL_CHAT = "https://api.zhizengzeng.com/v1/chat/completions"
API_URL_EMBED = "https://api.zhizengzeng.com/v1/embeddings"
OPENAI_API_KEY = "sk-zk249ad191da5206f1ca5d3e7e6fc9bc527683091ad9c28b"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}"
}

# 您的提示词模板
DRUG_PROMPT_TMPL = (
    "You are a distinguished clinical pharmacologist and expert in cheminformatics. Your objective is to generate a comprehensive, machine-readable description for a drug class, identified by its ATC-3 code. "
    "This description must bridge its clinical application with its underlying pharmacological properties, enabling an AI model to learn complex drug-drug,drug-procedures and drug-disease relationships."
    "For the medication with ATC level-3 code: \"{atc3_code}\" (representative NDC: \"{ndc_code}\")\n\n"
    "Synthesize the following information into a single continuous paragraph of formal medical text.Never use quotation marks for any terms, names, or codes. "
    "Do not use bullet points, headings, or numbered lists. If a specific point (e.g., common procedures) is not clinically relevant or applicable, omit it from the description to maintain a natural flow."
    "Embed all medical codes naturally within the text immediately after the relevant term, using the formats 'ICD-9-CM: [code]' and 'ATC-3: [code]'.\n"
    "1.State Anatomical Main Group (ATC Level 1): State the drug's broader anatomical main group (e.g., C for Cardiovascular system).\n"
    "2.Define Therapeutic Context (ATC Level 2): Describe the therapeutic subgroup (e.g., C07 for Beta blocking agents). "
    "List the typical diseases it treats (with ICD-9-CM codes) and associated clinical procedures (with ICD-9-CM procedure codes).\n"
    "3.Detail Pharmacological Profile and Interactions (ATC Levels 3-5): Detail the specific pharmacological class, its precise mechanism of action (MoA), "
    "common synergistic co-prescriptions (with ATC-3 codes), and critical drug-drug interactions (DDIs).\n"
    "For example, your output should look like this: "
    "'C07AB selective beta-blocking agents (ATC-3: C07AB) belong to the cardiovascular system (ATC-1: C) and are primarily indicated for hypertension (ICD-9-CM: 401.9) and angina (ICD-9-CM: 413.9). They reduce sympathetic drive by blocking cardiac beta-1 receptors…… Commonly co-prescribed with ACE inhibitors (ATC-3: C09AA)…… Concurrent use with calcium channel blockers (ATC-3: C08DA) should be avoided due to bradycardia risk……"
)


# =========================
# 2. 核心请求函数
# =========================
def get_single_description(atc_code: str, ndc_code: str) -> str:
    prompt = DRUG_PROMPT_TMPL.format(atc3_code=atc_code, ndc_code=ndc_code)
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,  # 严格保持 0.1，证明低温下的高稳定性
        "max_tokens": 512
    }
    r = requests.post(API_URL_CHAT, headers=HEADERS, json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def get_embeddings_batch(texts: list) -> list:
    data = {
        "model": "text-embedding-3-large",
        "input": texts,
        "dimensions": 512
    }
    r = requests.post(API_URL_EMBED, headers=HEADERS, json=data)
    r.raise_for_status()
    return [item["embedding"] for item in r.json()["data"]]


# =========================
# 3. 相似度与变异性计算工具
# =========================
def calculate_cosine_similarity(vec1, vec2):
    v1, v2 = np.array(vec1), np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def run_variability_experiment(atc_code, ndc_code, trials=5):
    print(f"\n🚀 开始针对 ATC: {atc_code} (NDC: {ndc_code}) 的 {trials} 次变异性测试...")

    # 1. 生成 N 次文本
    texts = []
    for i in range(trials):
        print(f"   正在生成第 {i + 1}/{trials} 次文本...")
        text = get_single_description(atc_code, ndc_code)
        texts.append(text)

    # 2. 获取这 N 次文本的向量
    print("   正在将文本批量转换为 Embedding...")
    embeddings = get_embeddings_batch(texts)

    # 3. 计算两两之间的相似度 (对于5次生成，共 5*4/2 = 10 个组合)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = []
    cosine_sims = []

    pairs = list(itertools.combinations(range(trials), 2))
    for i, j in pairs:
        # 计算 ROUGE-L (F1-score)
        score = scorer.score(texts[i], texts[j])
        rouge_scores.append(score['rougeL'].fmeasure)

        # 计算 余弦相似度
        sim = calculate_cosine_similarity(embeddings[i], embeddings[j])
        cosine_sims.append(sim)

    # 4. 统计与输出
    mean_rouge = np.mean(rouge_scores)
    var_rouge = np.var(rouge_scores)

    mean_cos = np.mean(cosine_sims)
    var_cos = np.var(cosine_sims)

    print("\n" + "=" * 40)
    print(f"📊 测试结果报告 (ATC: {atc_code})")
    print("=" * 40)
    print(f"ROUGE-L F1 平均相似度 : {mean_rouge:.4f} (方差: {var_rouge:.6f})")
    print(f"Embedding 平均余弦相似度: {mean_cos:.4f} (方差: {var_cos:.8f})")
    print("=" * 40)

    return mean_cos, var_cos


# =========================
# 4. 主函数：随机挑几个药测一下
# =========================
if __name__ == "__main__":
    # 随便写两三个您数据里的 ATC 码和对应的 NDC 进行测试
    test_cases = [
        ("A01A", "54569523500"),  # 示例: 换成您数据里实际的 ATC 和 NDC
        ("B05B", "00338008504"),
        ("C07A", "00085036207"),
        ("D03B", "00074231660"),
        ("R05D", "00456068801"),
        ("S01X", "17478006235"),
        ("V06D", "00338001702"),
        ("P01C", "63323087715"),
        ("M02A", "54569433200"),
        ("J04B", "49938010101"),
        ("L01E", "00078043815"),
        ("H01C", "55390016110"),
        ("G03F", "00078037742"),
        ("N07X", "61787006204")
    ]

    for atc, ndc in test_cases:
        run_variability_experiment(atc, ndc, trials=5)



# C:\ProgramData\anaconda3\envs\env_lamrec\python.exe D:\Code\LAMRec-RAGBK\preprocess\variability_test.py
# C:\Program Files\JetBrains\PyCharm 2024.2.0.1\plugins\python-ce\helpers\pycharm_display\datalore\display\supported_data_type.py:6: UserWarning: The NumPy module was reloaded (imported a second time). This can in some cases result in small but subtle issues and is discouraged.
#   import numpy
# C:\Users\LY\AppData\Roaming\Python\Python39\site-packages\nltk\metrics\association.py:26: UserWarning: A NumPy version >=1.22.4 and <2.3.0 is required for this version of SciPy (detected version 1.21.2)
#   from scipy.stats import fisher_exact
#
# 🚀 开始针对 ATC: A01A (NDC: 54569523500) 的 5 次变异性测试...
#    正在生成第 1/5 次文本...
#    正在生成第 2/5 次文本...
#    正在生成第 3/5 次文本...
#    正在生成第 4/5 次文本...
#    正在生成第 5/5 次文本...
#    正在将文本批量转换为 Embedding...
#
# ========================================
# 📊 测试结果报告 (ATC: A01A)
# ========================================
# ROUGE-L F1 平均相似度 : 0.7453 (方差: 0.005855)
# Embedding 平均余弦相似度: 0.9877 (方差: 0.00001655)
# ========================================
#
# 🚀 开始针对 ATC: B05B (NDC: 00338008504) 的 5 次变异性测试...
#    正在生成第 1/5 次文本...
#    正在生成第 2/5 次文本...
#    正在生成第 3/5 次文本...
#    正在生成第 4/5 次文本...
#    正在生成第 5/5 次文本...
#    正在将文本批量转换为 Embedding...
#
# ========================================
# 📊 测试结果报告 (ATC: B05B)
# ========================================
# ROUGE-L F1 平均相似度 : 0.5810 (方差: 0.013766)
# Embedding 平均余弦相似度: 0.9449 (方差: 0.00102493)
# ========================================
#
# 🚀 开始针对 ATC: C07A (NDC: 00085036207) 的 5 次变异性测试...
#    正在生成第 1/5 次文本...
#    正在生成第 2/5 次文本...
#    正在生成第 3/5 次文本...
#    正在生成第 4/5 次文本...
#    正在生成第 5/5 次文本...
#    正在将文本批量转换为 Embedding...
#
# ========================================
# 📊 测试结果报告 (ATC: C07A)
# ========================================
# ROUGE-L F1 平均相似度 : 0.6925 (方差: 0.008292)
# Embedding 平均余弦相似度: 0.9770 (方差: 0.00008168)
# ========================================
#
# 🚀 开始针对 ATC: D03B (NDC: 00074231660) 的 5 次变异性测试...
#    正在生成第 1/5 次文本...
#    正在生成第 2/5 次文本...
#    正在生成第 3/5 次文本...
#    正在生成第 4/5 次文本...
#    正在生成第 5/5 次文本...
#    正在将文本批量转换为 Embedding...
#
# ========================================
# 📊 测试结果报告 (ATC: D03B)
# ========================================
# ROUGE-L F1 平均相似度 : 0.7387 (方差: 0.014910)
# Embedding 平均余弦相似度: 0.9216 (方差: 0.00353268)
# ========================================
#
# 🚀 开始针对 ATC: R05D (NDC: 00456068801) 的 5 次变异性测试...
#    正在生成第 1/5 次文本...
#    正在生成第 2/5 次文本...
#    正在生成第 3/5 次文本...
#    正在生成第 4/5 次文本...
#    正在生成第 5/5 次文本...
#    正在将文本批量转换为 Embedding...
#
# ========================================
# 📊 测试结果报告 (ATC: R05D)
# ========================================
# ROUGE-L F1 平均相似度 : 0.8615 (方差: 0.010915)
# Embedding 平均余弦相似度: 0.9885 (方差: 0.00011724)
# ========================================
#
# 🚀 开始针对 ATC: S01X (NDC: 17478006235) 的 5 次变异性测试...
#    正在生成第 1/5 次文本...
#    正在生成第 2/5 次文本...
#    正在生成第 3/5 次文本...
#    正在生成第 4/5 次文本...
#    正在生成第 5/5 次文本...
#    正在将文本批量转换为 Embedding...
#
# ========================================
# 📊 测试结果报告 (ATC: S01X)
# ========================================
# ROUGE-L F1 平均相似度 : 0.4899 (方差: 0.003205)
# Embedding 平均余弦相似度: 0.9726 (方差: 0.00023619)
# ========================================
#
# 🚀 开始针对 ATC: V06D (NDC: 00338001702) 的 5 次变异性测试...
#    正在生成第 1/5 次文本...
#    正在生成第 2/5 次文本...
#    正在生成第 3/5 次文本...
#    正在生成第 4/5 次文本...
#    正在生成第 5/5 次文本...
#    正在将文本批量转换为 Embedding...
#
# ========================================
# 📊 测试结果报告 (ATC: V06D)
# ========================================
# ROUGE-L F1 平均相似度 : 0.6166 (方差: 0.009532)
# Embedding 平均余弦相似度: 0.9801 (方差: 0.00005637)
# ========================================
#
# 🚀 开始针对 ATC: P01C (NDC: 63323087715) 的 5 次变异性测试...
#    正在生成第 1/5 次文本...
#    正在生成第 2/5 次文本...
#    正在生成第 3/5 次文本...
#    正在生成第 4/5 次文本...
#    正在生成第 5/5 次文本...
#    正在将文本批量转换为 Embedding...
#
# ========================================
# 📊 测试结果报告 (ATC: P01C)
# ========================================
# ROUGE-L F1 平均相似度 : 0.4754 (方差: 0.008064)
# Embedding 平均余弦相似度: 0.8678 (方差: 0.00361061)
# ========================================
#
# 🚀 开始针对 ATC: M02A (NDC: 54569433200) 的 5 次变异性测试...
#    正在生成第 1/5 次文本...
#    正在生成第 2/5 次文本...
#    正在生成第 3/5 次文本...
#    正在生成第 4/5 次文本...
#    正在生成第 5/5 次文本...
#    正在将文本批量转换为 Embedding...
#
# ========================================
# 📊 测试结果报告 (ATC: M02A)
# ========================================
# ROUGE-L F1 平均相似度 : 0.8414 (方差: 0.003286)
# Embedding 平均余弦相似度: 0.9879 (方差: 0.00002409)
# ========================================
#
# 🚀 开始针对 ATC: J04B (NDC: 49938010101) 的 5 次变异性测试...
#    正在生成第 1/5 次文本...
#    正在生成第 2/5 次文本...
#    正在生成第 3/5 次文本...
#    正在生成第 4/5 次文本...
#    正在生成第 5/5 次文本...
#    正在将文本批量转换为 Embedding...
#
# ========================================
# 📊 测试结果报告 (ATC: J04B)
# ========================================
# ROUGE-L F1 平均相似度 : 0.7608 (方差: 0.005488)
# Embedding 平均余弦相似度: 0.9924 (方差: 0.00000887)
# ========================================
#
# 🚀 开始针对 ATC: L01E (NDC: 00078043815) 的 5 次变异性测试...
#    正在生成第 1/5 次文本...
#    正在生成第 2/5 次文本...
#    正在生成第 3/5 次文本...
#    正在生成第 4/5 次文本...
#    正在生成第 5/5 次文本...
#    正在将文本批量转换为 Embedding...
#
# ========================================
# 📊 测试结果报告 (ATC: L01E)
# ========================================
# ROUGE-L F1 平均相似度 : 0.7381 (方差: 0.013858)
# Embedding 平均余弦相似度: 0.9915 (方差: 0.00001638)
# ========================================
#
# 🚀 开始针对 ATC: H01C (NDC: 55390016110) 的 5 次变异性测试...
#    正在生成第 1/5 次文本...
#    正在生成第 2/5 次文本...
#    正在生成第 3/5 次文本...
#    正在生成第 4/5 次文本...
#    正在生成第 5/5 次文本...
#    正在将文本批量转换为 Embedding...
#
# ========================================
# 📊 测试结果报告 (ATC: H01C)
# ========================================
# ROUGE-L F1 平均相似度 : 0.5158 (方差: 0.030855)
# Embedding 平均余弦相似度: 0.8052 (方差: 0.01217334)
# ========================================
#
# 🚀 开始针对 ATC: G03F (NDC: 00078037742) 的 5 次变异性测试...
#    正在生成第 1/5 次文本...
#    正在生成第 2/5 次文本...
#    正在生成第 3/5 次文本...
#    正在生成第 4/5 次文本...
#    正在生成第 5/5 次文本...
#    正在将文本批量转换为 Embedding...
#
# ========================================
# 📊 测试结果报告 (ATC: G03F)
# ========================================
# ROUGE-L F1 平均相似度 : 0.5488 (方差: 0.001661)
# Embedding 平均余弦相似度: 0.9735 (方差: 0.00003344)
# ========================================
#
# 🚀 开始针对 ATC: N07X (NDC: 61787006204) 的 5 次变异性测试...
#    正在生成第 1/5 次文本...
#    正在生成第 2/5 次文本...
#    正在生成第 3/5 次文本...
#    正在生成第 4/5 次文本...
#    正在生成第 5/5 次文本...
#    正在将文本批量转换为 Embedding...
#
# ========================================
# 📊 测试结果报告 (ATC: N07X)
# ========================================
# ROUGE-L F1 平均相似度 : 0.6461 (方差: 0.019013)
# Embedding 平均余弦相似度: 0.9694 (方差: 0.00034922)
# ========================================
#
# Process finished with exit code 0