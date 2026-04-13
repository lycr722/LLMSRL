# custom_dataset.py

import dill
from pyhealth.datasets import SampleEHRDataset
from typing import List, Dict


class PreprocessedDataset(SampleEHRDataset):
    """
    最终版自定义数据集类。

    该版本严格遵循“一次就诊即一个样本”的规则，为 records_final.pkl 中的每一次就诊
    都生成一个对应的样本。对于患者的第一次就诊，其历史记录将为空列表。

    此版本修复了 `drugs_hist` 字段在空历史记录时列表嵌套层级不一致的问题。
    """

    def __init__(self, records_path: str, voc_path: str):

        print("正在从预处理的pickle文件加载数据 (最终版逻辑)...")

        with open(records_path, 'rb') as f:
            records = dill.load(f)

        with open(voc_path, 'rb') as f:
            voc = dill.load(f)

        self.voc = voc
        diag_voc = voc['diag_voc']
        med_voc = voc['med_voc']
        pro_voc = voc['pro_voc']

        samples = self._create_samples_from_records(records, diag_voc, med_voc, pro_voc)

        print(f"数据加载完成。共生成 {len(samples)} 个样本。")

        super().__init__(samples=samples, dataset_name="MIMIC3_Preprocessed", task_name="drug_recommendation")

    def _create_samples_from_records(self, records: List, diag_voc, med_voc, pro_voc) -> List[Dict]:
        samples = []
        patient_id_counter = 0

        for patient_visits in records:
            patient_id_counter += 1

            for visit_idx in range(len(patient_visits)):

                history_visits = patient_visits[:visit_idx]
                current_visit = patient_visits[visit_idx]

                conditions_hist = [[diag_voc.idx2word[idx] for idx in visit[0]] for visit in history_visits]
                procedures_hist = [[pro_voc.idx2word[idx] for idx in visit[1]] for visit in history_visits]
                drugs_hist = [[med_voc.idx2word[idx] for idx in visit[2]] for visit in history_visits]

                current_conditions_codes = [diag_voc.idx2word[idx] for idx in current_visit[0]]
                current_procedures_codes = [pro_voc.idx2word[idx] for idx in current_visit[1]]
                target_drugs = [med_voc.idx2word[idx] for idx in current_visit[2]]

                # +++ 核心修改点 +++
                # pyhealth要求所有样本的列表嵌套层级必须一致。
                # 对于第一次就诊，drugs_hist是[] (1层嵌套)，而后续是[[]] (2层嵌套)，这会导致错误。
                # 我们通过这个判断，确保即使是空历史，其结构也是2层嵌套的 [[]]。
                # 而 conditions 和 procedures 因为后面拼接了当前就诊，所以天然是2层嵌套，无需修改。
                if not drugs_hist:
                    drugs_hist = [[]]
                # ++++++++++++++++++

                sample = {
                    "patient_id": str(patient_id_counter),
                    "visit_id": f"{patient_id_counter}_{visit_idx}",
                    "conditions": conditions_hist + [current_conditions_codes],
                    "procedures": procedures_hist + [current_procedures_codes],
                    "drugs_hist": drugs_hist,
                    "drugs": target_drugs
                }
                samples.append(sample)

        return samples