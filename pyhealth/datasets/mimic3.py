import os
from typing import Dict
from collections import Counter
from typing import List
import pandas as pd
import numpy as np

from pyhealth.data import Event, Visit, Patient
from pyhealth.datasets import BaseEHRDataset
from pyhealth.datasets.utils import strptime


class MIMIC3Dataset(BaseEHRDataset):
    """
    修改后的MIMIC-III数据集加载类。

    集成了来自processing.py的特定数据处理逻辑，包括：
    - 筛选Top 2000的诊断编码。
    - 复杂的药物处理流程：NDC到ATC的映射，基于预定义列表的过滤，
      以及筛选Top 300的药物。
    """

    def __init__(self, root: str, tables: List[str], **kwargs):
        # 接收自定义文件路径
        self.custom_mapping_files = kwargs.pop("custom_mapping_files", None)
        self.med_structure_file = kwargs.pop("med_structure_file", None)
        super().__init__(root=root, tables=tables, **kwargs)

    def parse_basic_info(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """解析患者和就诊基础信息，此部分保持不变。"""
        patients_df = pd.read_csv(
            os.path.join(self.root, "PATIENTS.csv"),
            dtype={"SUBJECT_ID": str},
            nrows=1000 if self.dev else None,
        )
        admissions_df = pd.read_csv(
            os.path.join(self.root, "ADMISSIONS.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str},
        )
        df = pd.merge(patients_df, admissions_df, on="SUBJECT_ID", how="inner")
        df = df.sort_values(["SUBJECT_ID", "ADMITTIME", "DISCHTIME"], ascending=True)
        df_group = df.groupby("SUBJECT_ID")

        def basic_unit(p_id, p_info):
            patient = Patient(
                patient_id=p_id,
                birth_datetime=strptime(p_info["DOB"].values[0]),
                death_datetime=strptime(p_info["DOD_HOSP"].values[0]),
                gender=p_info["GENDER"].values[0],
                ethnicity=p_info["ETHNICITY"].values[0],
            )
            for v_id, v_info in p_info.groupby("HADM_ID"):
                visit = Visit(
                    visit_id=v_id,
                    patient_id=p_id,
                    encounter_time=strptime(v_info["ADMITTIME"].values[0]),
                    discharge_time=strptime(v_info["DISCHTIME"].values[0]),
                    discharge_status=v_info["HOSPITAL_EXPIRE_FLAG"].values[0],
                    insurance=v_info["INSURANCE"].values[0],
                    language=v_info["LANGUAGE"].values[0],
                    religion=v_info["RELIGION"].values[0],
                    marital_status=v_info["MARITAL_STATUS"].values[0],
                    ethnicity=v_info["ETHNICITY"].values[0],
                )
                patient.add_visit(visit)
            return patient

        # 使用.apply而不是.parallel_apply以获得更好的兼容性和调试
        processed_patients = df_group.apply(
            lambda x: basic_unit(x.SUBJECT_ID.unique()[0], x)
        )
        for pat in processed_patients:
            patients[pat.patient_id] = pat

        return patients

    def parse_diagnoses_icd(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """解析诊断信息，并筛选Top 2000最常见的诊断编码。"""
        table = "DIAGNOSES_ICD"
        self.code_vocs["conditions"] = "ICD9CM"
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "ICD9_CODE": str},
        )
        df = df[df["SUBJECT_ID"].isin(patients.keys())]
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"])

        # --- 新增逻辑: 筛选最常见的2000个ICD9编码 ---
        top_codes = df['ICD9_CODE'].value_counts().nlargest(2000).index
        df = df[df['ICD9_CODE'].isin(top_codes)]

        df = df.sort_values(["SUBJECT_ID", "HADM_ID", "SEQ_NUM"], ascending=True)
        group_df = df.groupby("SUBJECT_ID")

        def diagnosis_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("HADM_ID"):
                for code in v_info["ICD9_CODE"]:
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="ICD9CM",
                        visit_id=v_id,
                        patient_id=p_id,
                    )
                    events.append(event)
            return events

        group_df_applied = group_df.apply(
            lambda x: diagnosis_unit(x.SUBJECT_ID.unique()[0], x)
        )
        patients = self._add_events_to_patient_dict(patients, group_df_applied)
        return patients

    def parse_procedures_icd(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """解析手术信息，此部分保持不变。"""
        table = "PROCEDURES_ICD"
        self.code_vocs["procedures"] = "ICD9PROC"
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "ICD9_CODE": str},
        )
        df = df[df["SUBJECT_ID"].isin(patients.keys())]
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"])
        df = df.sort_values(["SUBJECT_ID", "HADM_ID", "SEQ_NUM"], ascending=True)
        group_df = df.groupby("SUBJECT_ID")

        def procedure_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("HADM_ID"):
                for code in v_info["ICD9_CODE"]:
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="ICD9PROC",
                        visit_id=v_id,
                        patient_id=p_id,
                    )
                    events.append(event)
            return events

        group_df_applied = group_df.apply(
            lambda x: procedure_unit(x.SUBJECT_ID.unique()[0], x)
        )
        patients = self._add_events_to_patient_dict(patients, group_df_applied)
        return patients

    def parse_prescriptions(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """解析药物信息，集成来自processing.py的完整处理流程。"""
        table = "PRESCRIPTIONS"
        self.code_vocs["drugs"] = "ATC"
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            low_memory=False,
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "NDC": str},
        )
        df = df[df["SUBJECT_ID"].isin(patients.keys())]

        # --- 步骤1: 清理和初步过滤 (来自 med_process) ---
        df.drop(index=df[df['NDC'] == '0'].index, axis=0, inplace=True)
        df.fillna(method='pad', inplace=True)
        df.dropna(subset=["SUBJECT_ID", "HADM_ID", "NDC"], inplace=True)
        df.drop_duplicates(inplace=True)

        # --- 步骤2: NDC -> ATC 映射 ---
        ndc2rxnorm = eval(open(self.custom_mapping_files["ndc2rxnorm"], "r").read())
        rxnorm2atc = pd.read_csv(self.custom_mapping_files["rxnorm2atc"])
        rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)

        df['RXCUI'] = df['NDC'].map(ndc2rxnorm)
        df = df.dropna(subset=['RXCUI'])
        df['RXCUI'] = pd.to_numeric(df['RXCUI'], errors='coerce')
        df = df.dropna(subset=['RXCUI'])
        df['RXCUI'] = df['RXCUI'].astype('int64')

        df = df.merge(rxnorm2atc[['RXCUI', 'ATC4']], on='RXCUI', how='inner')
        df = df.dropna(subset=['ATC4'])
        df['ATC'] = df['ATC4'].str[:4]

        # --- 步骤3: 药物编码过滤 ---
        if self.med_structure_file:
            med_structure = pd.read_pickle(self.med_structure_file)
            allowed_atcs = set(med_structure.keys())
            df = df[df['ATC'].isin(allowed_atcs)]

        top_atcs = df['ATC'].value_counts().nlargest(300).index
        df = df[df['ATC'].isin(top_atcs)]

        # --- 步骤4: 排序并创建Event对象 ---
        df = df.sort_values([
            "SUBJECT_ID", "HADM_ID", "STARTDATE", "ENDDATE"
        ], ascending=True)

        group_df = df.groupby("SUBJECT_ID")

        def prescription_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("HADM_ID"):
                for timestamp, code in zip(v_info["STARTDATE"], v_info["ATC"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="ATC",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp),
                    )
                    events.append(event)
            return events

        group_df_applied = group_df.apply(
            lambda x: prescription_unit(x.SUBJECT_ID.unique()[0], x)
        )
        patients = self._add_events_to_patient_dict(patients, group_df_applied)
        return patients