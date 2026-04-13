LLMSRL: LLM-enhanced synergetic representation learning for drug recommendation

LLM , MIMIC-III 1.4 / MIMIC-IV 2.2 , DDI , PyHealth , Multi-modal Clinical Representation  

📂 Project Structure
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
请将数据放置在：
mimic-iii-1.4/
mimic-iv-2.2/

📈 Evaluation
Jaccard
F1
Precision@k
DDI rate
Coverage
