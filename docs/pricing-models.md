# 定价模型说明

## 1. 支持的期权与模型分派

| 期权类型 | 模型 | 解析/数值 | 必需参数 |
| --- | --- | --- | --- |
| `european_call` / `european_put` | Black-Scholes-Merton | 闭式解 | spot, strike, maturity |
| `cash_or_nothing_call` / `cash_or_nothing_put` | Black-Scholes-Merton（数字期权） | 闭式解 | 上述 + `cash>0` |
| `american_call` / `american_put` | Cox-Ross-Rubinstein 二叉树 | 数值（后向归纳，含提前行权） | 上述 + `tree_steps∈[1,20000]` |

隐含波动率反解接口 `/api/v1/pricing/implied-vol` **仅支持欧式香草看涨/看跌**
（数字期权在波动率方向非单调、美式无闭式价格，均显式拒绝，返回
`INVALID_MODEL_PARAMS`）。

## 2. Black-Scholes-Merton

标的支付连续股息率 q：

```
d1 = [ ln(S/K) + (r - q + σ²/2) T ] / (σ√T)
d2 = d1 - σ√T
Call = S e^{-qT} N(d1) - K e^{-rT} N(d2)
Put  = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)
Digital-Call = C e^{-rT} N(d2)
Digital-Put  = C e^{-rT} N(-d2)
```

- `N(x)` 用标准库 `math.erfc` 计算（双精度机器精度），无第三方数值依赖。
- Greeks 全部为解析导数（数字期权的 gamma/vega/theta/rho 同样解析推导），
  测试以中心有限差分为解析基准对照（容差 2e-3 相对）。
- 约定：vega 以每 1.00 波动率计；theta 返回 `-∂V/∂T`（距离到期减少 1 年的
  价值变化）；T=0 返回内在价值（数字期权 ATM 取 `0.5·C`，即 Φ(0)=1/2 的
  到期极限），该点 Greeks 数学上退化，按 0 返回。

## 3. CRR 二叉树（美式）

```
u = e^{σ√Δt}, d = 1/u, p = (e^{(r-q)Δt} - d)/(u-d), 折现 e^{-rΔt}
每节点价值 = max(继续持有 e^{-rΔt}[p V_u + (1-p) V_d], 立即行权收益)
```

- 一维数组向后滚动，节点价格滚动生成；步数越大越接近 BSM 基准
  （无股息美式看涨不应提前行权，1000 步与欧式解析价偏差 < 0.005）。
- 美式 Greeks 用**固定步长**中心差分计算，保证同输入逐位可复现。
- 步数安全上限 20000，超限以 `INVALID_MODEL_PARAMS` 拒绝。

## 4. 隐含波动率

Brent 法（含反二次插值 + 割线 + 二分回退）在有根区间 `[1e-8, 5.0]` 内求解，
价格容差 1e-10，最多 100 次迭代。无解情形（目标价 ≤ 0、低于零波动下界、
高于 σ=500% 上界、或迭代不收敛）一律返回
`SOLVER_DID_NOT_CONVERGE`，绝不返回猜测值。

## 5. 解析基准（用于回归对照）

| 场景 | 解析值 |
| --- | --- |
| S=K=100, T=1, r=5%, q=0, σ=20% 欧式看涨 | 10.450583572185565 |
| 同上 欧式看跌 | 5.573526022256971 |
| 买卖权平价（含股息） | C − P = S e^{−qT} − K e^{−rT} |
| 数字 Call + Put | C · e^{−rT} |

平坦 20% 曲面（`tests` 中的 `FLAT-20PCT`）上的定价必须与上表一致到 1e-10。
