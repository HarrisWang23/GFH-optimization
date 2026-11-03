import pandas as pd
import numpy as np
import cvxpy as cp
import pymysql

# 建立数据库连接
conn = pymysql.connect(
    host='###',
    user='###',
    password='###',
    db='ch00'
)

# SQL 查询（仅提取 SZSE 的收盘价）
query = """
SELECT dt, ticker, last
FROM adjprc
WHERE exch = 'SZSE';
"""

df = pd.read_sql(query, conn)
print(df.head())
conn.close()


# 1. 转换为“日期 × 股票”的价格矩阵
price_df = df.pivot(index='dt', columns='ticker', values='last')

# 可选：去掉交易天数太少的股票（这里设置90%要求）
price_df = price_df.dropna(axis=1, thresh=0.9 * len(price_df))

# 2. 计算对数收益率（log return）
returns = np.log(price_df / price_df.shift(1)).dropna()

# 3. 计算协方差矩阵
cov_matrix = returns.cov()

# 查看结果
print("价格矩阵维度：", price_df.shape)
print("收益率矩阵维度：", returns.shape)
print("协方差矩阵维度：", cov_matrix.shape)
print(cov_matrix.iloc[:5, :5])  # 查看前 5×5 区块



mu = returns.mean().values                     # 平均收益率
Sigma = cov_matrix.values                      # 协方差矩阵
tickers = returns.columns.tolist()
n_stocks = len(tickers)

print(f"使用 {n_stocks} 支股票进行组合优化")

sigma2_max = 0.0005   # 可根据需求调整

w = cp.Variable(n_stocks)

objective = cp.Maximize(mu @ w)
constraints = [
    cp.quad_form(w, Sigma) <= sigma2_max,
    cp.sum(w) == 1,
    w >= 0
]

prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.MOSEK)

print("\n=== 最优组合结果 ===")
for i in range(n_stocks):
    if w.value[i] > 1e-4:
        print(f"{tickers[i]}: {w.value[i]:.4%}")

print(f"\n组合期望收益率: {mu @ w.value:.4%}")
print(f"组合方差: {w.value.T @ Sigma @ w.value:.6f}")
