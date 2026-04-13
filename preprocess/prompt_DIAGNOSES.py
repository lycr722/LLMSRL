import pandas as pd
import requests
import time
import csv
import os

# from contrast import prompt

'''
"你是一名临床药理学专家。您的任务是为给定的 ICD-9 诊断代码提供详细的临床相关描述。"
        "目标是生成一个描述，帮助机器学习模型理解该诊断对推荐药物组合的影响。"
        "对于 ICD-9 诊断代码\"{icd9_code}\"\n\n"
        "请用简洁的段落描述以下内容：\n"
        "1. 核心定义：简要定义诊断。\n"
        "2. 临床意义和严重程度：描述典型的临床表现、严重程度和常见的健康后果。提及该病症通常是常见病还是疑难杂症\n"
        "3. 常见并存病：列出与该诊断同时出现的常见并存疾病或病症。这对于理解病人为什么需要复杂的药物治疗方案至关重要。\n"
        "典型治疗目标和药物类别：这种疾病的主要治疗目标是什么（例如，缓解症状、治愈、减缓病情发展）？为实现这些目标，通常使用哪几类药物（如 "降压药"、"抗生素"、"抗凝药"）？\n\n"
        "请用一个连续的段落回答，不要换行或附加注释"。
        "3. Common Co-morbidities: List common co-existing diseases or conditions that frequently appear alongside this diagnosis. This is crucial for understanding why a patient might need a complex drug regimen.\n"
        "4. Typical Treatment Goals & Medication Classes: What are the primary goals of treatment for this condition (e.g., symptom relief, cure, slowing progression)? What general classes of drugs (e.g., 'antihypertensives', 'antibiotics', 'anticoagulants') are typically used to achieve these goals?\n\n"


ICD-9 代码 40301 指的是 "高血压性心脏病伴心力衰竭"，其特点是由于长期高血压导致心脏结构和功能发生变化，从而出现心力衰竭。
在临床上，患者可能会出现呼吸困难、疲劳、水肿和运动耐力下降等症状，这表明这是一种慢性疾病，如果处理不当，可能会导致严重的发病率和死亡率。
常见的健康后果包括左心室肥厚、心律失常以及心肌梗死和中风的风险增加。
与此诊断相关的并发症通常包括冠状动脉疾病、糖尿病、慢性肾病和心房颤动，
因此必须采用复杂的药物治疗方案来控制这些相互关联的问题。主要治疗目标集中在缓解症状、改善生活质量和延缓疾病进展上，通常通过使用降压药（如 ACE 抑制剂、β-受体阻滞剂、利尿剂）、心力衰竭药物（如醛固酮拮抗剂）等药物类别来实现，如果存在心房颤动，还可能使用抗凝药，这凸显了综合药物治疗方法的必要性。

"你是一名临床药理学专家。您的任务是为给定的 ICD-9 诊断代码提供详细的临床相关描述。"
        "目标是生成一个描述，帮助机器学习模型理解该诊断对推荐药物组合的影响。"
        "对于 ICD-9 诊断代码\"{icd9_code}\"\n\n"
        "请用简洁的段落描述以下内容：\n"
        "1. 核心定义：简要定义诊断。\n"
        "2. 临床意义和严重程度：描述典型的临床表现、严重程度和常见的健康后果。提及该疾病通常是常见病还是疑难病。\n"
        "3. 常见并存病：列出经常与该诊断同时出现的常见并存疾病或病症。这对于理解患者为何需要复杂的药物治疗至关重要。" "4.
        "关键程序：列出经常进行的常见手术或程序性干预。以及相关程序如何影响药物选择。\n"
        "5. 典型治疗目标和药物类别：该病症的主要治疗目标是什么？为了达到这些目标且基于以上几点分析，通常会使用哪些一线药物和辅助药物？\n\n"
        "请用一个连续的段落回答，不要换行或附加注释。

"你是一名临床药理学专家。您的任务是为给定的 ICD-9 诊断代码提供详细的临床相关描述。"
        "目标是生成一个描述，帮助机器学习模型理解该诊断对推荐药物组合的影响。"
        "对于 ICD-9 诊断代码\"{icd9_code}\"\n\n"
        "请用简洁的段落描述以下内容：\n"
        "1. 核心定义：简要定义诊断。\n"
        "2. 临床意义和严重程度：描述典型的临床表现、严重程度和常见的健康后果。提及该疾病通常是常见病还是疑难病。\n"
        "3. 常见并存病：列出经常与该诊断同时出现的常见并存疾病或病症（含ICD-9疾病编码）。这对于理解患者为何需要复杂的药物治疗方案至关重要。" "4.
        "4.主要禁忌症或注意事项：至关重要的是，由于可能导致病情恶化或产生严重副作用，患有这种诊断的患者一般应避免或极其谨慎地使用哪些特定药物或药物类别？（含药物ATC-3编码或NDC编码）"
        "5.常见相关程序：为了治疗或进一步诊断，该诊断的患者经常接受哪些内科或外科手术（含ICD-9手术编码）？这将诊断与潜在的行动联系起来。以及相关程序如何影响药物选择。\n"
        "6. 典型的治疗目标和药物类别：病情的主要治疗目标是什么？根据上述要点分析，通常使用哪些一线药物和辅助药物（含药物ATC-3编码或NDC编码）来实现这些目标？\n\n"
        "使用一个连续的段落，使用正式的医学语言。切勿使用项目符号或标题。酌情将实际代码（ATC-3、ICD-9-CM、ICD-9、NDC）自然嵌入文本中。"。

ICD-9 代码 40301 指的是 "高血压性心脏病伴心力衰竭"，其特点是长期高血压导致心力衰竭。典型的临床表现包括气短、乏力、水肿和心悸，通常会导致住院治疗。这种情况比较常见，尤其是在老年人中，因此在选择药物时，应优先考虑成熟的降压疗法。
常见的并发症包括慢性肾病、糖尿病和冠状动脉疾病，因此必须采用复杂的药物治疗方案来同时控制多种疾病。
主要禁忌症包括非甾体抗炎药（NSAIDs）和某些抗高血压药物，如严重哮喘患者使用的β-受体阻滞剂，因为这些药物可能会加重心力衰竭或导致严重的副作用。
常见的相关程序包括评估心脏功能的超声心动图检查和可能的评估冠状动脉疾病的心导管检查，这可能会影响药物的选择，需要仔细监测肾功能和电解质水平。
主要治疗目标是控制血压、缓解心衰症状和提高生活质量。一线药物通常包括血管紧张素转换酶（ACE）抑制剂、利尿剂和β-受体阻滞剂，而辅助疗法可能包括醛固酮拮抗剂和新药，如沙库比曲利/缬沙坦，并根据患者的总体健康状况和并发症进行调整。
'''

# API 设置 (请替换为您的KEY)
API_URL = "https://api.zhizengzeng.com/v1/chat/completions"
OPENAI_API_KEY = "sk-zk249ad191da5206f1ca5d3e7e6fc9bc527683091ad9c28b"  # <-- 请在这里填入您的 API Key

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + OPENAI_API_KEY
}


# ==============================================================================
# 1. 针对“诊断 (Diagnosis)” 的新 PROMPT
# ==============================================================================
def get_diagnosis_description(icd9_code):
    """
    使用为“诊断”优化的新Prompt调用大模型。
    这个Prompt引导模型从临床影响、共病和典型用药方向进行描述，
    为药物推荐提供更强的信号。
    """
    # prompt = (
    #     "You are an expert clinical pharmacologist. Your task is to provide a detailed and clinically relevant description for a given ICD-9 diagnosis code. "
    #     "The goal is to generate a description that helps a machine learning model understand the implications of this diagnosis for recommending a combination of medications. "
    #     "For the ICD-9 code: \"{icd9_code}\"\n\n"
    #     "Please provide a description that covers the following aspects in a concise paragraph:\n"
    #     # "1. Core Definition: Briefly define the diagnosis.\n"
    #     # "2. Clinical Implications & Severity: Describe the typical clinical manifestations, severity, and common health consequences. Mention whether the condition is usually a common or difficult disease.\n"
    #     "1. Core Definition: Briefly define the diagnosis.Describe the typical clinical manifestations.Mention whether the disease is usually common or difficult, how does this affect the choice of medication?\n"
    #     "2. Common Co-morbidities: List common co-existing diseases or conditions that frequently appear alongside this diagnosis. This is crucial for understanding why a patient might need a complex drug regimen."
    #     "3. Key Contraindications or Cautions: Crucially, what specific drugs or drug classes should generally be AVOIDED or used with extreme caution in patients with this diagnosis due to potential worsening of the condition or severe side effects?"
    #     "4. Commonly Associated Procedures: What medical or surgical procedures are frequently performed on patients with this diagnosis, either for treatment or further diagnosis? This links the diagnosis to potential actions.And how associated procedures affect medication choices.\n"
    #     "5. Typical treatment goals and drug classes: What are the primary treatment goals for the condition? What first-line and adjunctive medications are typically used to achieve these goals and based on the analysis of the above points?\n\n"
    #     "Please respond in a single continuous paragraph without any line breaks or additional commentary."
    # )
    # prompt = (
    #     "You are a clinical pharmacologist. Your task is to provide a detailed clinically relevant description for a given ICD-9 diagnosis code. The goal is to generate a description that helps a machine learning model understand the impact of the diagnosis on the recommended drug combination."
    #     # "The goal is to generate a description that helps a machine learning model understand the impact of the diagnosis on the recommended drug combination."
    #     "For ICD-9 diagnosis codes\"{icd9_code}\"\n\n"
    #     # "Please structure your description by following a logical clinical narrative. Begin by briefly defining the diagnosis. Following that, describe its typical clinical presentation, severity, common health consequences, and mention whether the disease is generally considered common or complex. Next, it is critical to detail the common co-existing conditions (including their ICD-9 codes) that often co-occur, as this helps explain the need for a complex medication regimen. Then, outline any major contraindications by specifying which drugs or drug classes (with their ATC-3 or NDC codes) should be avoided or used with extreme caution. Additionally, list commonly related medical or surgical procedures (with their ICD-9 procedure codes), explaining how these interventions influence medication choices. Finally, based on the preceding analysis, summarize the primary treatment objectives and identify the typical first-line and adjunctive medications (including their ATC-3 or NDC codes) used to achieve these goals."
    #     "1. Core definition: briefly define the diagnosis. \n"
    #     "2. CLINICAL IMPLICATIONS AND SERIOUSNESS: Describe the typical clinical presentation, severity, and common health consequences. Mention whether the disease is usually common or difficult. \n"
    #     "3. Common Co-existing Conditions: list the common co-existing diseases or conditions (with ICD-9 disease codes) that often co-occur with the diagnosis. This is critical to understanding why the patient requires a complex medication regimen.\n"
    #     "4. MAJOR CONTRAINDICATIONS OR CONTRAINDICATIONS: It is critical to know what specific drugs or classes of drugs should generally be avoided or used with extreme caution in patients with this diagnosis due to the potential for worsening of the condition or serious side effects? (with drug ATC-3 code or NDC code)\n"
    #     "5. COMMONLY RELATED PROCEDURES: What medical or surgical procedures are frequently performed on patients with this diagnosis for the purpose of treatment or further diagnosis (with ICD-9 procedure codes)? This links the diagnosis to potential actions. and how related procedures may affect medication choices. \n"
    #     "6. TYPICAL TREATMENT OBJECTIVES AND DRUG CATEGORIES: What are the primary treatment goals for the condition? Based on the bullet point analysis above, what first-line and adjunctive medications (with drug ATC-3 codes or NDC codes) are typically used to achieve these goals? \n\n"
    #     "Use one continuous paragraph using formal medical language. Do not use bullet points or headings. Naturally embed the actual code (ATC-3, ICD-9-CM, ICD-9, NDC) into the text as appropriate."
    # )
    # prompt = (
    #     "You are a clinical pharmacologist. Your task is to provide a detailed clinically relevant description for a given ICD-9-CM diagnosis code. The goal is to generate a description that helps a machine learning model understand the impact of the diagnosis on the recommended drug combination."
    #     "For ICD-9-CM diagnosis codes\"{icd9_code}\"\n\n"
    #     "Synthesize the following information into a single continuous paragraph of formal medical text. Do not use bullet points, headings, or numbered lists. If a specific point (e.g., common procedures) is not clinically relevant or applicable, omit it from the description to maintain a natural flow.Embed all medical codes naturally within the text immediately after the relevant term, using the formats 'ICD-9-CM: [code]' and 'ATC-3: [code]'.\n"
    #     "1. Core definition and typical clinical presentation. \n"
    #     "2. CLINICAL IMPLICATIONS AND SERIOUSNESS: Describe the typical clinical presentation, severity, and common health consequences. Mention whether the disease is usually common or difficult. \n"
    #     "3. Common Co-existing Conditions: list the common co-existing diseases or conditions (with ICD-9-CM disease codes) that often co-occur with the diagnosis. This is critical to understanding why the patient requires a complex medication regimen.\n"
    #     "4. MAJOR CONTRAINDICATIONS OR CONTRAINDICATIONS: It is critical to know what specific drugs or classes of drugs should generally be avoided or used with extreme caution in patients with this diagnosis due to the potential for worsening of the condition or serious side effects? (with drug ATC-3 code)\n"
    #     "5. COMMONLY RELATED PROCEDURES: What medical or surgical procedures are frequently performed on patients with this diagnosis for the purpose of treatment or further diagnosis (with ICD-9-CM procedure codes)? This links the diagnosis to potential actions. and how related procedures may affect medication choices. \n"
    #     "6. TYPICAL TREATMENT OBJECTIVES AND DRUG CATEGORIES: What are the primary treatment goals for the condition? Based on the bullet point analysis above, what first-line and adjunctive medications (with drug ATC-3 codes) are typically used to achieve these goals? \n\n"
    #     "For example, your output should look like this: 'Congestive heart failure (ICD-9-CM: 428.0) is a chronic condition characterized by the heart's inability to pump blood effectively, often leading to significant morbidity and frequently co-occurring with hypertension (ICD-9-CM: 401.9) and chronic kidney disease (ICD-9-CM: 585.9). Non-steroidal anti-inflammatory drugs (ATC-3: M01A) are generally contraindicated due to the risk of worsening fluid retention. Treatment focuses on reducing cardiac workload and managing symptoms, primarily using ACE inhibitors such as lisinopril (ATC-3: C09AA) and diuretics like furosemide (ATC-3: C03CA). Coronary artery bypass grafting (ICD-9-CM Proc: 36.1) may be indicated in severe cases.'"
    # )
    prompt = (
        "You are a senior clinician-scientist and biomedical informatician. Your task is to provide a detailed clinically relevant description for a given ICD-9-CM diagnosis code. The goal is to generate a description that helps a machine learning model understand the impact of the diagnosis on the recommended drug combination."
        "For ICD-9-CM diagnosis codes\"{icd9_code}\"\n\n"
        "Synthesize the following information into a single continuous paragraph of formal medical text.Never use quotation marks for any terms, names, or codes. Do not use bullet points, headings, or numbered lists. If a specific point (e.g., common procedures) is not clinically relevant or applicable, omit it from the description to maintain a natural flow.Embed all medical codes naturally within the text immediately after the relevant term, using the formats 'ICD-9-CM: [code]' and 'ATC-3: [code]'.\n"
        "1. Core Definition: Briefly define the diagnosis and its typical clinical impact."
        "2. Common co-existing diseases (with their ICD-9-CM diagnosis codes) that influence polypharmacy. "
        "3. Major Contraindications: Specify drugs or drug classes (with ATC-3 codes) that should be avoided or used with extreme caution.Crucially, explain why "
        "4. Relevant medical or surgical procedures (with ICD-9-CM procedure codes) commonly associated with this diagnosis. "
        "5. Treatment goals and typical drug classes: What are the primary treatment goals (e.g., symptom management, slowing progression, cure)? Based on these goals, what are the first-line and second-line drug classes (with ATC-3 codes) typically prescribed, and what is the reasoning?"
        "For example, your output should look like this: 'Type 2 diabetes mellitus (ICD-9-CM: 250.00) is a chronic metabolic disorder where the primary treatment goal is glycemic control to prevent long-term complications. This condition frequently co-exists with hypertension (ICD-9-CM: 401.9) and hyperlipidemia (ICD-9-CM: 272.4), necessitating complex polypharmacy. First-line therapy is typically metformin (ATC-3: A10BA) due to its efficacy and low hypoglycemia risk, while second-line agents like SGLT2 inhibitors (ATC-3: A10BK) are often added for their proven cardiovascular benefits. In advanced cases with peripheral vascular disease, procedures such as lower limb amputation (ICD-9-CM Proc: 84.1) may become necessary. A major contraindication involves avoiding thiazolidinediones (ATC-3: A10BG) in patients with congestive heart failure (ICD-9-CM: 428.0), as this drug class can cause significant fluid retention and exacerbate the condition.'"
    )
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt.format(icd9_code=icd9_code)}],
        "temperature": 0.5,  # 对于事实性描述，稍低的温度可能效果更好
        "stream": False
    }
    response = requests.post(API_URL, headers=HEADERS, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ==============================================================================
# 2. 针对“治疗过程 (Procedure)” 的新 PROMPT
# ==============================================================================
def get_procedure_description(icd9_code):
    """
    使用为“治疗过程”优化的新Prompt调用大模型。
    这个Prompt引导模型关注操作的适应症（连接诊断）和术后用药，
    直接为药物推荐提供上下文。
    """
    # prompt = (
    #     "You are an expert clinical informatician. Your task is to provide a detailed and clinically relevant description for a given ICD-9 procedure code. "
    #     "The goal is to generate a description that helps a machine learning model understand the context and purpose of this procedure, especially its relationship to subsequent medication needs. "
    #     "For the ICD-9 procedure code: \"{icd9_code}\"\n\n"
    #     "Please provide a description that covers the following aspects in a concise paragraph:\n"
    #     "1. Procedure Definition: Briefly describe what the procedure is.\n"
    #     "2. Primary Indications: For what primary diagnoses or conditions is this procedure typically performed? This directly links the procedure back to the patient's problems.\n"
    #     "3. Purpose of Procedure: What is the main goal of this procedure (e.g., diagnostic, therapeutic, palliative)?\n"
    #     "4. Common Post-Procedure Medications: What types of medications are commonly prescribed after this procedure? Consider medications for pain management, infection prevention, management of the underlying condition, or preventing complications (e.g., 'analgesics', 'prophylactic antibiotics', 'thromboprophylaxis').\n\n"
    #     "Please respond in a single continuous paragraph without any line breaks or additional commentary."
    # )
    # prompt = (
    # "You are an expert clinical informatician. Your task is to provide a detailed and clinically relevant description for a given ICD-9 procedure code. "
    # "The goal is to generate a description that helps a machine learning model understand the context and purpose of this procedure, especially its relationship to subsequent medication needs. "
    # "For the ICD-9 procedure code: \"{icd9_code}\"\n\n"
    # "Please provide a description that covers the following aspects in a concise paragraph:\n"
    # "1. Procedure Definition & Purpose: Briefly describe the procedure and its main goal (e.g., diagnostic, therapeutic).\n"
    # "2. Primary Indications (Diagnoses): For what primary ICD-9 diagnoses is this procedure typically performed?\n"
    # "3. Common Post-Procedure Medications: What classes of medications(with drug NDC/ATC-3 codes) are commonly prescribed immediately following this procedure for recovery and management?\n"
    # "4. Key Contraindications or Cautions: Are there specific drugs or drug classes(with drug NDC/ATC-3 codes) that are often contraindicated or must be stopped prior to or immediately after this procedure?\n\n"
    # "Use one continuous paragraph using formal medical language. Do not use bullet points or headings. Embed the actual codes (ATC-3, ICD-9-CM, ICD-9, NDC) naturally in the text as appropriate."
    # )
    # prompt = (
    # "You are an expert clinical informatician and surgical pharmacologist. Your task is to provide a detailed and clinically relevant description for a given ICD-9-CM procedure code.The goal is to generate a description that helps a machine learning model understand the context and purpose of this procedure, especially its relationship to subsequent medication needs. "
    # "For the ICD-9-CM procedure code: \"{icd9_code}\"\n\n"
    # "Synthesize the following information into a single continuous paragraph of formal medical text. Do not use bullet points, headings, or numbered lists. If a specific point (e.g., common procedures) is not clinically relevant or applicable, omit it from the description to maintain a natural flow.Embed all medical codes naturally within the text immediately after the relevant term, using the formats 'ICD-9-CM: [code]' and 'ATC-3: [code]'.\n"
    # "1. Procedure Definition & Purpose: Briefly describe the procedure and its main goal (e.g., diagnostic, therapeutic).\n"
    # "2. Primary Indications (Diagnoses): For what primary ICD-9-CM diagnoses is this procedure typically performed?\n"
    # "3. Common Post-Procedure Medications: What classes of medications(with drug ATC-3 codes) are commonly prescribed immediately following this procedure for recovery and management?\n"
    # "4. Key Contraindications or Cautions: Are there specific drugs or drug classes(with drug ATC-3 codes) that are often contraindicated or must be stopped prior to or immediately after this procedure?\n\n"
    # "For example, your output should look like this: 'Coronary artery bypass grafting (ICD-9-CM: 36.1) is a surgical procedure performed to restore blood flow to the heart muscle, primarily indicated for severe coronary artery disease (ICD-9-CM: 414.01). Post-procedure, patients are commonly managed with antiplatelet agents like aspirin (ATC-3: B01AC) and beta-blockers such as metoprolol (ATC-3: C07AB) to prevent graft occlusion and manage cardiac risk. It is critical to discontinue anticoagulants like warfarin (ATC-3: B01AA) prior to surgery to minimize bleeding risk.'"
    # )
    prompt = (
        "You are an expert clinical informatician and surgical pharmacologist. Your task is to provide a detailed and clinically relevant description for a given ICD-9-CM procedure code.The goal is to generate a description that helps a machine learning model understand the context and purpose of this procedure, especially its relationship to subsequent medication needs. "
        "For the ICD-9-CM procedure code: \"{icd9_code}\"\n\n"
        "Synthesize the following information into a single continuous paragraph of formal medical text.Never use quotation marks for any terms, names, or codes. Do not use bullet points, headings, or numbered lists. If a specific point (e.g., common procedures) is not clinically relevant or applicable, omit it from the description to maintain a natural flow.Embed all medical codes naturally within the text immediately after the relevant term, using the formats 'ICD-9-CM: [code]' and 'ATC-3: [code]'.\n"
        "1. Procedure Definition & Purpose: Briefly describe the procedure and its main goal (e.g., diagnostic, therapeutic).\n"
        "2. Primary Indications (Diagnoses): For what primary ICD-9-CM diagnoses is this procedure typically performed?\n"
        "3. Pre-procedure: What drug classes (with ATC-3 codes) often need to be held or adjusted before the procedure and why (e.g., 'discontinuation of anticoagulants (ATC-3: B01AA) to mitigate bleeding risk')?"
        "4. Post-procedure: What are the common short-term medications (with ATC-3 codes) prescribed immediately after, and for what purpose (e.g., 'for prophylaxis, pain management, or recovery support')?"
        "5. Key Contraindications or Cautions: Are there specific drugs or drug classes(with drug ATC-3 codes) that are often contraindicated or must be stopped prior to or immediately after this procedure?\n\n"
        "For example, your output should look like this: 'This procedure is primarily indicated for symptomatic cholelithiasis (ICD-9-CM: 574.20) and acute cholecystitis (ICD-9-CM: 575.11). A critical pre-procedure step involves the discontinuation of anticoagulants (ATC-3: B01AA) and antiplatelet agents (ATC-3: B01AC) several days prior to surgery; this is crucial to mitigate the risk of intraoperative and postoperative hemorrhage. Immediately post-procedure, patients are commonly prescribed opioid analgesics (ATC-3: N02A) for pain management and may receive a short course of first-generation cephalosporins (ATC-3: J01DB) for surgical site infection prophylaxis.'"
    )
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt.format(icd9_code=icd9_code)}],
        "temperature": 0.5,
        "stream": False
    }
    response = requests.post(API_URL, headers=HEADERS, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ==============================================================================
# 3. 主逻辑
# ==============================================================================
def process_codes(code_type='diagnosis'):
    """
    主处理函数。
    :param code_type: 'diagnosis' 或 'procedure'
    """
    if code_type == 'diagnosis':
        input_csv = "DIAGNOSES_ICD.csv"
        output_csv = "DIAGNOSES_Description_ID.csv"
        api_function = get_diagnosis_description
        print("🚀 Starting to process DIAGNOSIS codes...")
    elif code_type == 'procedure':
        input_csv = "mimic3_procedures_mapping.csv"
        output_csv = "PROCEDURES_Description_ID.csv"
        api_function = get_procedure_description
        print("🚀 Starting to process PROCEDURE codes...")
    else:
        print("❌ Invalid code_type. Please choose 'diagnosis' or 'procedure'.")
        return

    # 检查输入文件是否存在
    if not os.path.exists(input_csv):
        print(f"❌ Input file not found: {input_csv}")
        return

    # 读取原始 ICD9 代码
    df = pd.read_csv(input_csv, dtype={"ICD9_CODE": str})
    icd9_codes_all = sorted(set(df["ICD9_CODE"].dropna()))

    # 读取已完成的代码（如果有）
    completed_codes = set()
    # 第一次运行时创建文件头
    if not os.path.exists(output_csv):
        with open(output_csv, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["ICD9_CODE", "DESCRIPTION"])
    else:
        try:
            existing_df = pd.read_csv(output_csv, dtype={"ICD9_CODE": str})
            if "ICD9_CODE" in existing_df.columns:
                completed_codes = set(existing_df["ICD9_CODE"].dropna())
        except Exception as e:
            print(f"⚠️ Warning: Could not read existing output file. Starting from scratch. Error: {e}")

    # 过滤出未完成的代码
    pending_codes = [code for code in icd9_codes_all if code not in completed_codes]
    print(
        f"Total unique codes: {len(icd9_codes_all)}, Completed: {len(completed_codes)}, Pending: {len(pending_codes)}")

    if not pending_codes:
        print("✅ All codes have been processed. Exiting.")
        return

    # 以 append 模式继续写入
    with open(output_csv, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for idx, code in enumerate(pending_codes):
            # 防止空代码导致错误
            if not code or pd.isna(code):
                continue

            print(f"Processing [{idx + 1}/{len(pending_codes)}]: {code}")
            try:
                description = api_function(code)
                print(f"  -> Got description: {description[:80]}...")
                writer.writerow([code, description])
            except Exception as e:
                print(f"  ❌ Error on code {code}: {e}")
                writer.writerow([code, "ERROR"])

            # 尊重API的速率限制
            time.sleep(2)

    print(f"✅ Processing finished for {code_type}.")


def single(icd9_code):
    try:
        description = get_diagnosis_description(icd9_code)
        print(f"\n🩺 ICD-9 Code {icd9_code} 描述如下：\n")
        print(description)
    except Exception as e:
        print(f"❌ 请求失败：{e}")


if __name__ == "__main__":
    # icd = "40301"
    # single(icd)

    # 您想处理哪种类型的代码？ 'diagnosis' 或 'procedure'
    # =======================================================
    # target_code_type = 'diagnosis'
    target_code_type = 'procedure'
    # =======================================================

    process_codes(target_code_type)