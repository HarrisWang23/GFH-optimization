import os
import pandas as pd
from sqlalchemy import create_engine, text

# === 配置 ===
csv_folder = r"C:\Users\USER\Desktop\共享\csv"         # 修改为你的文件夹路径
mapping_path = r"C:\Users\USER\Desktop\mapping.csv" # mapping 表路径
base_column = r"C:\Users\USER\Desktop\共享\csv\close.csv"      # 用哪个文件的 ticker 当作统一格式（如 close.csv）

host = '###'
port = 3306
user = '###'
password = '###'
database = 'gjmh'
table_name = 'data'

engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4")

# === 读取 ticker 映射表 ===
mapping_df = pd.read_csv(mapping_path)

# === 所有文件名（不含隐藏文件）===
all_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv") and not f.startswith("~")]

# === 将每个文件转为 long 格式并标准化 ticker ===
merged_df = None
i=1
for file in all_files:
    path = os.path.join(csv_folder, file)
    indicator = os.path.splitext(file)[0]  # 文件名作为列名（如 close）
    
    df = pd.read_csv(path)
    df_long = df.melt(id_vars=["date"], var_name="ticker", value_name=indicator)
    
    # 检查是否需要映射
    need_mapping = df_long["ticker"].isin(mapping_df["abs_ticker"]).any()

    if need_mapping:
        df_long = df_long.merge(mapping_df, how="left", left_on="ticker", right_on="abs_ticker")
        df_long["ticker"] = df_long["close_ticker"]
        df_long.drop(columns=["abs_ticker", "close_ticker"], inplace=True)

    
    df_long["date"] = df_long["date"].astype(str)
    
    if merged_df is None:
        merged_df = df_long
    else:
        merged_df = pd.merge(merged_df, df_long, on=["date", "ticker"], how="outer")
    print(i)
    i+=1

# === 写入数据库 ===
with engine.connect() as conn:
    conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))

merged_df.to_sql(name=table_name, con=engine, if_exists="replace", index=False)

print(f"✅ 已成功将 {len(all_files)} 个文件合并至表 [{table_name}]，包含列：{merged_df.columns.tolist()}")
