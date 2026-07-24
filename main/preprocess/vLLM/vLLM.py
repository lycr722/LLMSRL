import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from vllm import LLM, SamplingParams

model_id = "BioMistral/BioMistral-7B"

# 论文中提到的生成参数 [cite: 364, 365]
sampling_params = SamplingParams(
    temperature=0.5, 
    max_tokens=512
)

# 初始化 LLM，占用一部分显存用于 KV Cache
llm = LLM(
    model="BioMistral/BioMistral-7B", 
    dtype="half", 
    gpu_memory_utilization=0.9,
    download_dir="/hy-tmp/huggingface_cache"  # 显式指定下载目录
)

# 假设这里是您遍历 EHR 数据集提取出来的不同 Prompt 列表
prompts = [
    "[INST] Generate description for ICD-9-CM: 453.40 [/INST]",
    "[INST] Generate description for ATC-3: B01A [/INST]",
    "[INST] Generate description for ICD-9-CM Proc: 36.10 [/INST]"
]

# 极速批量推理
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}\n---")