LLMSRL: LLM-based Safe Drug Recommendation with RAG-enhanced Clinical Semantics
📌 Overview
LLMSRL（Large Language Model for Safe Drug Recommendation with RAG-based Knowledge）是一个结合：

LLM（大语言模型）

RAG（Retrieval-Augmented Generation）

MIMIC-III / MIMIC-IV 临床数据

药物相互作用（DDI）图

PyHealth 框架

的药物推荐研究项目。

本项目旨在通过结构化 EHR 数据 + 医疗知识库 + LLM 语义增强，实现更安全、更可解释的药物推荐。

🚀 Features
RAG-based Clinical Embeddings  
使用 LLM（如 GPT / LLaMA）生成诊断、手术、药物的语义向量。

DDI-aware Drug Recommendation  
结合 DDI 图（如 ddi_adj_final_131.csv）减少危险组合。

Multi-modal Clinical Representation  
诊断、手术、药物 embedding 融合。

PyHealth Integration  
使用 PyHealth 的 dataset / model / metrics 体系。

可复现的训练脚本  
包括 run_mimic3.py、trainer.py、trainerLogDrug.py。

📂 Project Structure
代码
LAMRec-RAGBK/
│
├── preprocess/         # 数据预处理、embedding 生成、RAG prompt
├── models/             # 模型结构、embedding 文件、DDI 图
├── pyhealth/           # PyHealth 框架（本地修改版）
├── agent/              # RAG agent 相关代码
├── output/             # 模型输出
├── run_mimic3.py       # 主训练脚本
├── trainer.py          # 训练器
├── util.py             # 工具函数
└── README.md
🛠 Installation
1. Clone the repository
代码
git clone https://github.com/lycr722/LLMSRL.git
cd LLMSRL
2. Install dependencies
代码
pip install -r requirements.txt
如果你需要，我可以帮你自动生成 requirements.txt。

📊 Data Preparation
本项目使用：

MIMIC-III 1.4

MIMIC-IV 2.2

请将数据放置在：

代码
mimic-iii-1.4/
mimic-iv-2.2/
⚠️ 注意：MIMIC 数据受许可保护，不能上传到 GitHub。

🧠 Training
运行主训练脚本：

代码
python run_mimic3.py
或使用 Tri 模型：

代码
python run_mimic3Tri.py
📈 Evaluation
训练完成后，评估指标包括：

Jaccard

F1

Precision@k

DDI rate

Coverage
