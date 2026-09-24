# 期权定价与波动率曲面服务（本地独立运行）

一套可完全离线运行的期权定价服务：波动率与行情全部来自本地曲面数据，
同一市场状态在不同调用路径（同步 HTTP / 异步任务 / 直接调用）下结论一致、
可复现；非法输入、曲面越界、数值不收敛一律**拒绝并返回可区分的失败原因**，
绝不返回近似值或默认值。

- 纯标准库实现数值部分（`math.erf` 正态分布、CRR 二叉树、二分法），无需 numpy/scipy
- FastAPI + Pydantic，`uv run uvicorn main:app` 即可启动
- Python >= 3.14

## 1. 定价模型

| 期权类型 | 方法 | 说明 |
| --- | --- | --- |
| `vanilla_call` / `vanilla_put` | Black–Scholes–Merton 闭式解 | 连续股息率 q |
| `digital_call` / `digital_put` | BSM 现金或无（cash-or-nothing）闭式解 | 收益为固定 `payout` |
| `american_put` | Cox–Ross–Rubinstein 二叉树 | 步数倍增至相对变化 < 容差；到期前可提前行权 |
| `target_price`（欧式香草） | 隐含波动率二分求解 | 反解市场价格对应的 IV |

数值失败策略（均抛出异常，不返回近似结果）：

- `SolverNonConvergenceError`（HTTP 422，`solver_non_convergence`）
  - 美式期权：步数预算 `max_steps` 用尽仍未达到 `convergence_tol`；
  - 隐含波动率：目标价超出无套利价格区间（`InvalidParameterError`，
    错误信息给出可行价格上下界），或二分迭代预算用尽。
- 所有底层参数（S/K/T/σ/payout 必须 > 0，q ≥ 0，CRR 风险中性概率须在 (0,1)）
  在模型层与引擎层双重校验，拒绝码 `invalid_parameter`。

解析基准对照见 `tests/test_pricing_benchmark.py`：

- Hull《期权、期货及其他衍生产品》例：S=42, K=40, T=0.5, r=0.10, σ=0.20
  → call ≈ **4.7594**、put ≈ **0.8086**；
- 看涨看跌平价 `c + K e^-rT = p + S e^-qT`（误差 0）；
- 数字期权恒等式 `digital_call + digital_put = e^-rT`（误差 0）；
- 美式看跌（S=50,K=52,T=2,r=5%,σ=22.63%）收敛值 ≈ **5.540**，
  且内在价值 < 美式 ≤ 上界、不低于欧式看跌；
- IV 往返：IV(BS 价格) 恢复 σ（绝对误差 < 1e-7），重新定价误差 < 1e-8。

## 2. 波动率曲面与版本规则

- 曲面为**不可变快照**（`VolSurfaceSnapshot` 为 frozen dataclass）：
  到期日/行权价严格递增网格 + 双线性插值；
- 默认**禁止外插**：查询点落在 `[最小到期日, 最大到期日] ×
  [最小行权价, 最大行权价]` 之外时抛出 `SurfaceOutOfRangeError`
  （HTTP 422，`surface_out_of_range`），与“非法参数”区分；
  曲面发布时可设置 `allow_extrapolation=true` 启用线性外插；
- 每次 `POST /surfaces/{id}/versions` 生成一个**单调递增新版本号**，
  旧版本永不修改、永不删除，可永久回放；
- 定价请求可用 `version` 绑定历史版本；不传则在处理开始时锁定当时最新版本，
  并在响应/任务状态中返回 `surface_version`（任务失败时该版本号同样保留）；
- 启动时自动载入 `data/surfaces/*.json`（命名 `<id>_v<n>.json`，
  同 id 按文件名顺序发布为 v1、v2…）。当前内置：
  - `default` v1（S=100,r=5%）、v2（S=102,r=4.5%,q=1%）——用于版本回放对照；
  - `tech` v1（S=200，允许外插）。

**并发发布规则**：发布在 `SurfaceStore` 锁内“构建全新快照 → 一次性安装”，
读者要么看到完整旧版本、要么看到完整新版本，不存在新旧网格混杂或部分可见。
`tests/test_concurrency.py::test_concurrent_publish_never_mixes_versions`
以 4 发布线程 + 4 读取线程持续 0.8s 压测，校验每个被读取到的快照网格内部完全自洽。

## 3. 并发处理规则

- 异步任务在有界 `ThreadPoolExecutor`（默认 4 线程）中执行，
  **长计算不阻塞事件循环，也不阻塞其他请求**；
- 任务状态机：`pending → running → succeeded | failed | cancelled`，
  状态转换由每把任务锁保护；
- 确定性：所有定价器都是 `(请求, 曲面快照)` 的纯函数；
  `PricingService` 按 `(surface_id, version, 规范化请求哈希)` 缓存结果，
  相同输入 + 同一曲面版本在任意路径、任意并发下结果严格相同
  （`cached:true` 表示命中缓存）；
- 取消是协作式的：任务轮询 `threading.Event`（求解迭代边界检查），
  `POST /jobs/{id}/cancel` 后进入 `cancelled`（码 `job_cancelled`）；
- 单个任务失败/取消只影响自身：工作线程捕获全部异常并落库终态，
  服务继续对外服务（`/health` 与后续定价不受影响）。

## 4. HTTP 接口

| 方法 & 路径 | 作用 |
| --- | --- |
| `GET /health` | 健康检查 |
| `POST /pricing/price` | 同步定价 |
| `POST /surfaces/{id}/versions` | 发布新曲面版本 |
| `GET /surfaces/{id}/versions` | 列出全部版本 |
| `GET /surfaces/{id}/versions/{v}` | 查询指定版本元数据 |
| `GET /surfaces/{id}/versions/{v}/vol?expiry=&strike=` | 查询插值波动率 |
| `POST /jobs` | 提交异步定价任务（202，返回 job_id） |
| `GET /jobs/{id}` | 查询任务状态/结果/失败原因 |
| `POST /jobs/{id}/cancel` | 取消任务 |

错误响应统一格式：

```json
{"error": "surface_out_of_range", "message": "expiry 99.0 outside ...", "detail": {}}
```

| error 码 | 含义 | HTTP |
| --- | --- | --- |
| `invalid_parameter` | 参数非法 / pydantic 校验失败 / IV 无可行解 | 422 |
| `surface_out_of_range` | 查询点在曲面支撑之外且禁止外插 | 422 |
| `solver_non_convergence` | 数值求解未收敛 | 422 |
| `surface_not_found` / `job_not_found` | 曲面/版本/任务不存在 | 404 |
| `job_cancelled` | 任务被取消 | 任务体内 |
| `internal_error` | 未预期错误（不会击垮工作线程） | 500 |

### 使用示例

```bash
uv run uvicorn main:app --port 8000

# 绑定历史 v1 复现已发布结论
curl -s -X POST localhost:8000/pricing/price -H 'Content-Type: application/json' -d '{
  "surface_id":"default","version":1,
  "option_type":"vanilla_call","expiry":1,"strike":100}'

# 越界 -> 422 surface_out_of_range（不返回任何价格）
curl -s -X POST localhost:8000/pricing/price -H 'Content-Type: application/json' -d '{
  "option_type":"vanilla_call","expiry":99,"strike":100}'

# 异步美式期权
curl -s -X POST localhost:8000/jobs -H 'Content-Type: application/json' -d '{
  "option_type":"american_put","expiry":1,"strike":100,
  "steps":500,"max_steps":4000,"convergence_tol":1e-4}'
```

## 5. 验证方法

```bash
uv run pytest                       # 全部 22 个用例
uv run pytest tests/test_pricing_benchmark.py -v   # 解析基准对照
uv run pytest tests/test_validation.py -v          # 非法/越界/不收敛
uv run pytest tests/test_concurrency.py -v         # 并发发布/版本回放
uv run pytest tests/test_api.py -v                 # HTTP 契约/取消
```

测试日志（stdout 与 caplog 均有）对每个用例打印：

```
CASE=version_replay | INPUTS={'versions': [1, 2, 3]} | SURFACE_VERSION=[1,2,3]
  | CRITERION=each version reproduces its own price; sync and async paths identical...
```

即**输入参数、所用曲面版本、判定依据**三要素，便于审计与复现。

## 6. 目录结构

```
option_engine/
  errors.py     # 可区分失败原因的异常层级
  models.py     # Pydantic 请求/响应模型
  surface.py    # 不可变快照、双线性插值、版本化存储（原子发布）
  pricing.py    # BSM/数字闭式解、CRR 美式、隐含波动率
  service.py    # 版本绑定 + 规范化结果缓存（跨路径一致性）
  jobs.py       # 线程池异步任务、状态机、协作取消
  data.py       # 本地 JSON 曲面载入
  api.py        # FastAPI 路由与错误映射
data/surfaces/  # 本地曲面数据（default v1/v2、tech v1）
tests/          # 基准/非法参数/并发/HTTP 四组测试
```
