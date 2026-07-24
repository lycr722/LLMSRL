import os
import csv
import pickle
from tqdm import tqdm

# 强制使用国内镜像源，防止下载配置文件超时
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from vllm import LLM, SamplingParams

# =========================================================
# 基础配置 (适配 RTX 3060 12GB)
# =========================================================
# MODEL_ID = "TheBloke/BioMistral-7B-AWQ"
MODEL_ID = "BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM"
# 将这里的路径替换为您目标机器上具体的绝对路径
MODEL_ID = "/data/models/BioMistral-AWQ/snapshots/6739b645fb6a30dd9029c06b0bb477a47736648d"
# 原来是: model_id = "BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM"
# 现在改为目标机器上的绝对路径：
# local_model_path = "/data/models/BioMistral-AWQ"
# =========================================================
# 输入输出文件 (更新命名体系)
# =========================================================
DRUG_INPUT_CSV = "drug_code2index.csv"
NDC_ATC_MAP_CSV = "ndc_atc_map_level3.csv"
DRUG_OUTPUT_CSV = "BioMistral_Drug_Descriptions_ID.csv"

VOCAB_PKL = "official_vocabs.pkl"
DIAG_OUTPUT_CSV = "BioMistral_DIAGNOSES_Description_ID.csv"

PROC_INPUT_CSV = "mimic3_procedures_mapping.csv"
PROC_OUTPUT_CSV = "BioMistral_PROCEDURES_Description_ID.csv"

# =========================================================
# 系统指令 (融入 Prompt 中)
# =========================================================
SYSTEM_INSTRUCTION = (
    "You are a precise clinical biomedical writing assistant. "
    "Follow the user instruction exactly. "
    "Generate formal, clinically grounded, machine-readable medical text. "
    "Do not use bullet points, headings, numbered lists, or quotation marks unless explicitly requested. "
    "Keep the output as one continuous paragraph.\n\n"
)

# =========================================================
# Prompt 模板 (加入 [INST] 标签)
# =========================================================
DRUG_PROMPT_TMPL = (
        "[INST] " + SYSTEM_INSTRUCTION +
        "You are a distinguished clinical pharmacologist and expert in cheminformatics. "
        "Your objective is to generate a comprehensive, machine-readable description for a drug class, "
        "identified by its ATC-3 code. This description must bridge its clinical application with its "
        "underlying pharmacological properties, enabling an AI model to learn complex drug-drug, "
        "drug-procedure, and drug-disease relationships. "
        "For the medication with ATC level-3 code: {atc3_code} "
        "(representative NDC: {ndc_code}).\n\n"
        "Synthesize the following information into a single continuous paragraph of formal medical text. "
        "Never use quotation marks for any terms, names, or codes. "
        "Do not use bullet points, headings, or numbered lists. "
        "If a specific point is not clinically relevant or applicable, omit it to maintain a natural flow. "
        "Embed all medical codes naturally within the text immediately after the relevant term, using the formats "
        "ICD-9-CM: [code] and ATC-3: [code].\n"
        "1. State Anatomical Main Group (ATC Level 1): State the drug's broader anatomical main group.\n"
        "2. Define Therapeutic Context (ATC Level 2): Describe the therapeutic subgroup. "
        "List the typical diseases it treats with ICD-9-CM diagnosis codes and associated clinical procedures "
        "with ICD-9-CM procedure codes.\n"
        "3. Detail Pharmacological Profile and Interactions (ATC Levels 3-5): "
        "Detail the specific pharmacological class, its precise mechanism of action, common synergistic "
        "co-prescriptions with ATC-3 codes, and critical drug-drug interactions. [/INST]"
)

DIAGNOSIS_PROMPT_TMPL = (
        "[INST] " + SYSTEM_INSTRUCTION +
        "You are a senior clinician-scientist and biomedical informatician. "
        "Your task is to provide a detailed clinically relevant description for a given ICD-9-CM diagnosis code. "
        "The goal is to generate a description that helps a machine learning model understand the impact of "
        "the diagnosis on the recommended drug combination. "
        "For ICD-9-CM diagnosis code: {icd9_code}\n\n"
        "Synthesize the following information into a single continuous paragraph of formal medical text. "
        "Never use quotation marks for any terms, names, or codes. "
        "Do not use bullet points, headings, or numbered lists. "
        "If a specific point is not clinically relevant or applicable, omit it from the description to maintain a natural flow. "
        "Embed all medical codes naturally within the text immediately after the relevant term, using the formats "
        "ICD-9-CM: [code] and ATC-3: [code].\n"
        "1. Core Definition: Briefly define the diagnosis and its typical clinical impact.\n"
        "2. Common co-existing diseases with their ICD-9-CM diagnosis codes that influence polypharmacy.\n"
        "3. Major Contraindications: Specify drugs or drug classes with ATC-3 codes that should be avoided "
        "or used with extreme caution, and explain why.\n"
        "4. Relevant medical or surgical procedures with ICD-9-CM procedure codes commonly associated with this diagnosis.\n"
        "5. Treatment goals and typical drug classes: What are the primary treatment goals? "
        "What first-line and second-line drug classes with ATC-3 codes are typically prescribed? [/INST]"
)

PROCEDURE_PROMPT_TMPL = (
        "[INST] " + SYSTEM_INSTRUCTION +
        "You are an expert clinical informatician and surgical pharmacologist. "
        "Your task is to provide a detailed and clinically relevant description for a given ICD-9-CM procedure code. "
        "The goal is to generate a description that helps a machine learning model understand the context and purpose "
        "of this procedure, especially its relationship to subsequent medication needs. "
        "For the ICD-9-CM procedure code: {icd9_code}\n\n"
        "Synthesize the following information into a single continuous paragraph of formal medical text. "
        "Never use quotation marks for any terms, names, or codes. "
        "Do not use bullet points, headings, or numbered lists. "
        "Embed all medical codes naturally using ICD-9-CM: [code] and ATC-3: [code].\n"
        "1. Procedure definition and purpose.\n"
        "2. Primary indications with ICD-9-CM diagnosis codes.\n"
        "3. Pre-procedure medication adjustments with ATC-3 codes.\n"
        "4. Post-procedure medications with ATC-3 codes.\n"
        "5. Key contraindications. [/INST]"
)


# =========================================================
# 通用工具函数
# =========================================================
def clean_text(text: str) -> str:
    if text is None:
        return ""
    return " ".join(str(text).replace("\r", " ").replace("\n", " ").split())


def ensure_csv_header(output_csv: str, header: list):
    if not os.path.exists(output_csv):
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def read_completed_codes(output_csv: str, key_col: str) -> set:
    completed = set()
    if not os.path.exists(output_csv):
        return completed
    try:
        with open(output_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                value = row.get(key_col, "")
                if value is not None:
                    value = str(value).strip()
                    if value:
                        completed.add(value)
    except Exception as e:
        print(f"⚠️ 读取已完成文件失败 {output_csv}: {e}")
    return completed


def read_csv_column(file_path: str, column_name: str) -> list:
    values = []
    with open(file_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if column_name not in (reader.fieldnames or []):
            raise ValueError(f"{file_path} 中不存在列: {column_name}")
        for row in reader:
            value = row.get(column_name, "")
            if value is None:
                continue
            value = str(value).strip()
            if value:
                values.append(value)
    return values


def build_ndc_map(file_path: str) -> dict:
    ndc_map = {}
    if not os.path.exists(file_path):
        return ndc_map
    with open(file_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            atc = str(row.get("ATC", "")).strip()
            ndc = str(row.get("NDC", "")).strip()
            if atc and ndc:
                ndc_map[atc] = ndc
    return ndc_map


def unique_preserve_order(items: list) -> list:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# =========================================================
# 批处理生成核心逻辑
# =========================================================
def process_drug_descriptions(llm, sampling_params):
    print("\n🚀 Processing DRUG codes...")
    if not os.path.exists(DRUG_INPUT_CSV):
        return

    atc_codes_all = unique_preserve_order(read_csv_column(DRUG_INPUT_CSV, "code"))
    ndc_map = build_ndc_map(NDC_ATC_MAP_CSV)
    completed = read_completed_codes(DRUG_OUTPUT_CSV, "ATC")
    pending = [c for c in atc_codes_all if c not in completed]

    print(f"Total: {len(atc_codes_all)}, Completed: {len(completed)}, Pending: {len(pending)}")
    if not pending:
        return

    ensure_csv_header(DRUG_OUTPUT_CSV, ["ATC", "NDC", "DESCRIPTION"])

    # 1. 构建 Prompt 列表
    prompts = [DRUG_PROMPT_TMPL.format(atc3_code=c, ndc_code=ndc_map.get(c, "Unknown-NDC")) for c in pending]

    # 2. vLLM 并发推理
    print("Generating...")
    outputs = llm.generate(prompts, sampling_params)

    # 3. 保存结果 (vLLM 输出顺序与输入列表严格一致)
    with open(DRUG_OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for code, output in zip(pending, outputs):
            ndc = ndc_map.get(code, "Unknown-NDC")
            desc = clean_text(output.outputs[0].text)
            writer.writerow([code, ndc, desc])

    print(f"✅ Saved to '{DRUG_OUTPUT_CSV}'")


def process_diagnosis_descriptions(llm, sampling_params):
    print("\n🚀 Processing DIAGNOSIS codes...")
    if not os.path.exists(VOCAB_PKL):
        return

    with open(VOCAB_PKL, "rb") as f:
        vocabs = pickle.load(f)

    diag_codes_all = sorted(
        set([str(x).strip() for x in vocabs["diagnoses"] if x is not None and str(x).lower() != "nan"]))
    completed = read_completed_codes(DIAG_OUTPUT_CSV, "ICD9_CODE")
    pending = [c for c in diag_codes_all if c not in completed]

    print(f"Total: {len(diag_codes_all)}, Completed: {len(completed)}, Pending: {len(pending)}")
    if not pending:
        return

    ensure_csv_header(DIAG_OUTPUT_CSV, ["ICD9_CODE", "DESCRIPTION"])

    prompts = [DIAGNOSIS_PROMPT_TMPL.format(icd9_code=c) for c in pending]

    print("Generating...")
    outputs = llm.generate(prompts, sampling_params)

    with open(DIAG_OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for code, output in zip(pending, outputs):
            desc = clean_text(output.outputs[0].text)
            writer.writerow([code, desc])

    print(f"✅ Saved to '{DIAG_OUTPUT_CSV}'")


def process_procedure_descriptions(llm, sampling_params):
    print("\n🚀 Processing PROCEDURE codes...")
    if not os.path.exists(PROC_INPUT_CSV):
        return

    proc_codes_all = sorted(set(read_csv_column(PROC_INPUT_CSV, "ICD9_CODE")))
    completed = read_completed_codes(PROC_OUTPUT_CSV, "ICD9_CODE")
    pending = [c for c in proc_codes_all if c not in completed]

    print(f"Total: {len(proc_codes_all)}, Completed: {len(completed)}, Pending: {len(pending)}")
    if not pending:
        return

    ensure_csv_header(PROC_OUTPUT_CSV, ["ICD9_CODE", "DESCRIPTION"])

    prompts = [PROCEDURE_PROMPT_TMPL.format(icd9_code=c) for c in pending]

    print("Generating...")
    outputs = llm.generate(prompts, sampling_params)

    with open(PROC_OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for code, output in zip(pending, outputs):
            desc = clean_text(output.outputs[0].text)
            writer.writerow([code, desc])

    print(f"✅ Saved to '{PROC_OUTPUT_CSV}'")


# =========================================================
# 主函数
# =========================================================
def main():
    print("====================================================")
    print(f"Starting VLLM batch generation with {MODEL_ID}")
    print("====================================================")

    # 全局初始化一次 LLM 模型，放置于主存中，后续处理直接调用
    llm = LLM(
        model=MODEL_ID,
        quantization="awq",
        dtype="half",
        gpu_memory_utilization=0.9
    )

    # 统一定义参数 (依据您论文设定) [cite: 364, 365]
    sampling_params = SamplingParams(
        temperature=0.5,
        max_tokens=800  # 稍微放宽上限，防止文本截断
    )

    process_drug_descriptions(llm, sampling_params)
    process_diagnosis_descriptions(llm, sampling_params)
    process_procedure_descriptions(llm, sampling_params)

    print("\n🎉 All BioMistral tasks completed!")


if __name__ == "__main__":
    main()