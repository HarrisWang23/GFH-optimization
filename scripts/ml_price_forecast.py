import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine

# --------------------
# 1. 数据库连接（替换成你的）
# --------------------
engine = create_engine("mysql+pymysql://###:###@###:3306/ch00")  # 修改为你自己的信息
ticker = '000001'

# --------------------
# 2. 从数据库读取该股票最近61天数据
# --------------------
sql = f"""
SELECT dt, ticker, last
FROM (
    SELECT dt, ticker, last
    FROM adjprc
    WHERE ticker = '{ticker}'
    ORDER BY dt DESC
    LIMIT 61
) AS recent
ORDER BY dt ASC;
"""

start_time = time.time()
df = pd.read_sql(sql, engine)
print(df.tail())
print(f"[读取SQL数据] 用时：{time.time() - start_time:.2f} 秒")

# --------------------
# 3. 归一化与构造输入/真实值
# --------------------
assert len(df) == 61, "数据不足 61 天"
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df['last'].values.reshape(-1, 1))
X_input = scaled[:-1].reshape(1, 60, 1)         # 倒数第61天到第2天
y_true = scaled[-1]                             # 倒数第1天（真实）

# --------------------
# 4. 转为张量
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_input = torch.tensor(X_input, dtype=torch.float32).to(device)

# --------------------
# 5. 定义 Transformer 模型
# --------------------
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, seq_len=60, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):  # x: (batch, seq_len, 1)
        x = self.input_linear(x) + self.pos_encoding
        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)
        x = self.transformer(x)
        x = x[-1]  # (batch, d_model)
        return self.fc_out(x)  # (batch, 1)

# --------------------
# 6. 初始化空模型并推理
# --------------------
model = TimeSeriesTransformer().to(device)
model.eval()

with torch.no_grad():
    y_pred = model(X_input).cpu().numpy()[0][0]

# --------------------
# 7. 反归一化并输出结果
# --------------------
# 反归一化
y_pred_real = scaler.inverse_transform(np.array([[y_pred]]))[0][0]
y_true_real = scaler.inverse_transform(np.array([[y_true]] if np.isscalar(y_true) else y_true.reshape(1, -1)))[0][0]


print(f"\n[预测完成] 用时：{time.time() - start_time:.2f} 秒")
print(f"预测值（真实价格）: {y_pred_real:.2f}")
print(f"实际值（真实价格）: {y_true_real:.2f}")
