pip install transformers accelerate bitsandbytes vllm --default-timeout=1000 -i https://pypi.tuna.tsinghua.edu.cn/simple


将这个 zip 文件下载并传输到您的目标机器（或本地环境）后，建议按照以下方式调用，以避免因跨系统（如 Linux 到 Windows）解压导致 Hugging Face 软链接失效的问题：

将文件解压到目标机器的任意目录，例如 /data/models/BioMistral-AWQ。

进入解压后的文件夹，一层层点进去，找到包含真实实体文件的那个长哈希目录，即：
/data/models/BioMistral-AWQ/snapshots/6739b645fb6a30dd9029c06b0bb477a47736648d/

将您脚本中的 MODEL_ID 直接指向这个绝对路径：

Python
# 将这里的路径替换为您目标机器上具体的绝对路径
MODEL_ID = "/data/models/BioMistral-AWQ/snapshots/6739b645fb6a30dd9029c06b0bb477a47736648d"

llm = LLM(
    model=MODEL_ID, 
    quantization="awq", 
    dtype="half", 
    gpu_memory_utilization=0.9
)
通过这种直接指定底层 snapshot 文件夹绝对路径的方式，您可以完全脱离外网环境，在任何符合显存要求的机器上稳定地运行这些批量生成任务。接下来，您只需要按原计划将生成的文本利用 text-embedding-3-large 转化为向量，即可放入您的 LLMSRL 框架中进行重训并提取 Jaccard、F1 等对比指标了。