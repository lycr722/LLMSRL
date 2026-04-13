# import pandas as pd
# import numpy as np
#
# # 1. 加载数据
# # 假设文件在当前目录下
# drug_map = pd.read_csv('drug_code2index.csv')
# # ddi_A_final 第一列是药物代码，设为 index 以便检索
# ddi_full = pd.read_csv('ddi_A_final.csv', index_col=0)
#
# # 2. 提取目标代码列表（按 index 排序）
# target_codes = drug_map.sort_values('index')['code'].tolist()
#
# # 3. 构建新的邻接矩阵
# # 我们需要确保 target_codes 都在 ddi_full 的坐标中
# # 如果某些代码在 ddi_full 中不存在，则补 0
# available_codes = [code for code in target_codes if code in ddi_full.index]
# missing_codes = [code for code in target_codes if code not in ddi_full.index]
#
# if len(missing_codes) > 0:
#     print(f"警告：以下药物代码在原始 DDI 矩阵中未找到，将填充为 0: {missing_codes}")
#
# # 使用 reindex 进行行列重排和对齐
# # fill_value=0 处理缺失的代码
# ddi_adj_131 = ddi_full.reindex(index=target_codes, columns=target_codes, fill_value=0)
#
# # 4. 转换为纯数值矩阵（去除代码标签）
# adj_matrix = ddi_adj_131.values
#
# # 5. 保存结果
# # 保存为不带表头和索引的 CSV
# pd.DataFrame(adj_matrix).to_csv('ddi_adj_final_131.csv', index=False, header=False)
#
# print("处理完成！生成文件：ddi_adj_final_131.csv")
# print(f"矩阵形状: {adj_matrix.shape}")





import dill
import numpy as np
import pandas as pd

# 1. 加载 pkl 文件
# 注意：你的文件是通过 dill 序列化的，所以必须用 dill.load
with open('ddi_A_final.pkl', 'rb') as f:
    ddi_data = dill.load(f)

# 2. 检查数据类型并转换为 DataFrame
# 如果 ddi_data 是 numpy 数组
if isinstance(ddi_data, np.ndarray):
    df = pd.DataFrame(ddi_data)
else:
    # 如果是字典或其他格式，尝试直接转换
    df = pd.DataFrame(ddi_data)

# 3. 导出为 csv
df.to_csv('ddi_A_final.csv', index=False, header=False)

print(f"转换完成！矩阵形状为: {df.shape}")