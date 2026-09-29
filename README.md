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
| `american_put` | Cox–Ross–Rubinstein 二叉树 | 见下方“离散精度控制” |
| `target_price`（欧式香草） | 隐含波动率二分求解 | 反解市场价格对应的 IV |

### 离散精度控制（美式期权）

美式看跌提供两种**可区分**的定价模式，响应里的 `pricing_mode` 一眼可辨，并通过
`steps_used` 报告实际使用的步数：

- `pricing_mode = "fixed_resolution"`（固定分辨率）：请求传
  `resolution = N`，引擎**恰好**用 N 步 CRR 树算一次，不做任何收敛搜索、
  不替换步数。结果是 `(请求, 曲面版本)` 的确定性函数，同一曲面版本反复跑
  逐位相同，适合复现“固定期数/固定步长”的第三方报价。
  - **允许范围**：`N ∈ [1, 2000]`（常量
    `resolution.MIN_RESOLUTION` / `MAX_RESOLUTION`，经 `models` 再导出），
    区间内**每个整数都受理并实际计算**，**边界值 1 与 2000 同样受理**；
  - **允许的类型只有两种**：JSON 整数（Python `int`，且
    `isinstance(x, int)` 为真）或省略该字段（`null`，走自动收敛）。
    规则全部集中在 `option_engine/resolution.py::validate_resolution`
    这一个函数里，三个入口共用、逐条调用，表现完全一致；
  - **类型非法一律拒绝，绝不换算成任何步数**（不会出现 `true` 当 1 步、
    `1.0` 当 1 步、`"100"` 当 100 步这类静默强转）：
    - 布尔 `true` / `false` → 拒绝（`bool` 虽是 `int` 子类也单独拦截），
      原因中注明 “got boolean …”；
    - 浮点 `1.0` / `1.5` / `0.999`（含“整数值浮点”）→ 拒绝，
      原因中注明实际收到的浮点值；
    - 字符串 `"100"` / `"1.0"` / `"true"`（任何字符串）→ 拒绝，
      不做任何数字解析，原因中注明实际收到的字符串；
    - 其它非整数类型（数组、对象等）→ 拒绝；
  - **区间外的整数继续拒绝**：0、负数、>2000 均拒绝，
    **不会在尚未到上限时就报“算不出来”**（2000 照常计算，2001 才拒绝）；
  - 上述拒绝全部以 `invalid_parameter`（HTTP 422）返回，错误消息会点明
    **字段名 `resolution`** 与**不可受理的具体原因（类型/区间）**，
    且**绝不返回任何价格**；
  - `resolution` 只对 `american_put` 合法，欧式闭式解传该字段会收到
    `invalid_parameter`；
  - 固定模式下 `steps / convergence_tol / max_steps` 不参与计算；
    `detail` 为 `{"steps_used": N, "converged": null, "resolution": N}`。
- **三个入口的一致行为**（同一套规则、同一种判定）：
  - 同步 `POST /pricing/price`：非法类型在请求体校验阶段返回
    **422 `invalid_parameter`**，`detail.errors[].loc` 末位为
    `resolution`，`message` 含字段名与原因；
  - 异步 `POST /jobs`：非法类型在**提交时**即返回 422
    （不会创建任务，也不会进入 pending/running）；
  - 批量 `POST /pricing/price/batch`：条目的原始值先保留、不做整批强转；
    **某一条 `resolution` 非法时只让这一条失败**（该条
    `status="failed"`、`error_code="invalid_parameter"`、自带字段级原因、
    `result=null`），**其余条目照常定价返回**，不会因一条非法值让整批
    422，也不会拖住或影响其它条目/其它请求；合法条目的结果与单独调用
    逐字段一致（价格、`vol_used`、`pricing_mode`、`steps_used`）。
  - 引擎层（`PricingService._compute`）另有一道同样的防御性校验：即使
    请求对象绕过模型层（如 `model_construct`），脏值也进不了二叉树。
- `pricing_mode = "auto_converged"`（自动收敛，默认）：不传 `resolution`
  时保持原有行为——从 `steps` 起倍增直到相对变化 < `convergence_tol`，
  `steps_used` 为实际收敛步数；预算 `max_steps` 用尽仍不收敛则抛出
  `SolverNonConvergenceError`，绝不返回近似值。
- 欧式（含数字期权、IV 反解）为闭式解，`pricing_mode = "closed_form"`、
  `steps_used = null`。

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
  （HTTP 422，`surface_out_of_range`），与“非法参数”区分。
- **外插授权（双重条件）**：不再由随曲面写死的布尔开关决定。
  1. 曲面版本在**发布时**通过 `extrapolation_limits`
     （`min_expiry/max_expiry/min_strike/max_strike`，必须包住网格支撑）
     声明可外插范围；未声明的版本**永远不能外插**，调用方请求也没用。
  2. 调用方在**每次请求**显式传 `allow_extrapolation: true` 才会外插；
     即使曲面声明了范围，不请求外插仍按越界拒绝。
  - 实际查询点必须落在声明范围之内（含边界）；超出声明范围照样
    `surface_out_of_range`，不会退化回拒绝以外的任何默认行为。
  - 恰在网格支撑边界上的点算插值，不算外插；恰在声明范围边界上的点
    允许外插。
- **外插审计（从不静默）**：只要实际用到了外插，`PriceResponse` 中
  `extrapolated = true`，并带 `extrapolation` 块：
  `expiry_outside / strike_outside`（哪条轴出界）、
  `expiry_overhang / strike_overhang`（距网格支撑的绝对距离）、
  `expiry_overhang_fraction / strike_overhang_fraction`
  （相对声明外插带宽的占比，0–1）。插值命中时 `extrapolated = false`、
  `extrapolation = null`。`GET .../vol` 查询端点同样支持
  `?allow_extrapolation=true` 并返回相同审计字段。
- 授权与范围**随版本固化**：绑定历史版本回放时，该版本当时声明的范围
  （或“未声明、不可外插”）原样生效，后续发布新版本不改变历史结论。
- 每次 `POST /surfaces/{id}/versions` 生成一个**单调递增新版本号**，
  旧版本永不修改、永不删除，可永久回放；
- 定价请求可用 `version` 绑定历史版本；不传则在处理开始时锁定当时最新版本，
  并在响应/任务状态中返回 `surface_version`（任务失败时该版本号同样保留）；
- 启动时自动载入 `data/surfaces/*.json`（命名 `<id>_v<n>.json`，
  同 id 按文件名顺序发布为 v1、v2…）。当前内置：
  - `default` v1（S=100,r=5%）、v2（S=102,r=4.5%,q=1%）——未声明外插范围，
    用于版本回放对照；
  - `tech` v1（S=200）——声明可外插范围
    expiry ∈ [0.1, 3.0]、strike ∈ [150, 260]。

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

## 4. 批量定价（每日组合 / 回测批处理）

`POST /pricing/price/batch` 一次提交一整批请求（例如一天的组合）：

- **整批绑定同一个曲面版本**：`surface_id` 与可选 `version` 在批次上只给一次；
  不传版本时在批次开始时锁定当时最新版本。版本/曲面不存在时整批以
  `surface_not_found`（404）失败（此时没有任何可定价的条目）。
- **逐条独立并发**：每条 leg 在专用有界线程池（默认最多 8 线程，按条目数
  缩减）上执行。某一条失败或算得慢**不影响其他条目、也不拖住其他请求**
  （同步 `/pricing/price`、异步任务与批量调用互不抢占）。
- **单条失败可单独归因**：响应恒为 HTTP 200，按提交下标 `index` 逐条返回
  `status: succeeded | failed`；失败条带独立的
  `error_code / error_message`（`surface_out_of_range`、
  `solver_non_convergence`、`invalid_parameter` 等），且 `result = null`，
  不拖垮整批。请求体结构非法（如空 `items`）仍在入口返回 422。
  - **`resolution` 类型/区间非法同样按单条失败处理**：例如某一条传
    `true`、`1.0` 或 `"100"`，只有这一条返回
    `status="failed" / error_code="invalid_parameter"` 且消息注明
    `resolution` 字段与具体原因，其它条目不受影响（批次本身仍为 200）。
    这与单独调用同步入口收到的失败码/原因完全一致（单独调用表现为
    HTTP 422，批量内表现为该条 failed）。
- **与单条调用完全一致**：每条 leg 在内部被构造成与独立调用完全相同的
  `PricingRequest`（含批次绑定的版本）并走同一计算与结果缓存路径，
  因此同一版本下批量每条价格、`vol_used`、`pricing_mode`、`steps_used`、
  外插审计均与单独调用严格相同。

请求/响应示例：

```json
// POST /pricing/price/batch
{
  "surface_id": "default", "version": 1,
  "items": [
    {"option_type": "vanilla_call", "expiry": 1.0, "strike": 100},
    {"option_type": "american_put", "expiry": 1.0, "strike": 100,
     "resolution": 252},
    {"option_type": "vanilla_call", "expiry": 99.0, "strike": 100}
  ]
}
// 200
{
  "surface_id": "default", "surface_version": 1,
  "results": [
    {"index": 0, "status": "succeeded", "result": {"price": "...", ...}},
    {"index": 1, "status": "succeeded",
     "result": {"pricing_mode": "fixed_resolution", "steps_used": 252, ...}},
    {"index": 2, "status": "failed",
     "error_code": "surface_out_of_range", "error_message": "...",
     "result": null}
  ]
}
```

## 5. HTTP 接口

| 方法 & 路径 | 作用 |
| --- | --- |
| `GET /health` | 健康检查 |
| `POST /pricing/price` | 同步定价（支持 `resolution` / `allow_extrapolation`） |
| `POST /pricing/price/batch` | 批量定价（单版本绑定，逐条独立成败） |
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
| `invalid_parameter` | 参数非法 / pydantic 校验失败（含 `resolution` 布尔、浮点、字符串、越界）/ IV 无可行解 | 422 |
| `surface_out_of_range` | 越界且未获外插授权：未请求外插 / 曲面版本未声明范围 / 超出声明范围 | 422 |
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

# 固定分辨率：恰好 252 步的确定性美式价（用于复现第三方“固定期数”报价）
curl -s -X POST localhost:8000/pricing/price -H 'Content-Type: application/json' -d '{
  "surface_id":"default","version":1,"option_type":"american_put",
  "expiry":1,"strike":100,"resolution":252}'

# 非法分辨率类型：true / 1.0 / "100" 一律 422 invalid_parameter，
# 不会被换算成 1 / 1 / 100 步，也不会返回任何价格
curl -s -i -X POST localhost:8000/pricing/price -H 'Content-Type: application/json' -d '{
  "surface_id":"default","version":1,"option_type":"american_put",
  "expiry":1,"strike":100,"resolution":true}'
# -> HTTP/1.1 422 ; body.error = "invalid_parameter" ;
#    message 点明字段 resolution 与 “got boolean true”，
#    detail.errors[0].loc 末位为 "resolution"

# 区间外整数同样 422（边界 1 与 2000 受理，0 与 2001 拒绝）
curl -s -X POST localhost:8000/pricing/price -H 'Content-Type: application/json' -d '{
  "surface_id":"default","version":1,"option_type":"american_put",
  "expiry":1,"strike":100,"resolution":0}'

# 批量：非法分辨率只失败对应条目，其它条目照常定价（整批仍 200）
curl -s -X POST localhost:8000/pricing/price/batch -H 'Content-Type: application/json' -d '{
  "surface_id":"default","version":1,"items":[
    {"option_type":"american_put","expiry":1,"strike":100,"resolution":137},
    {"option_type":"american_put","expiry":1,"strike":100,"resolution":true},
    {"option_type":"vanilla_put","expiry":1,"strike":100}]}'
# -> results[0] succeeded(fixed_resolution,137)
#    results[1] failed invalid_parameter（原因注明 boolean，result=null）
#    results[2] succeeded(closed_form)

# 在曲面声明的范围内显式请求外插（响应带 extrapolation 审计块）
curl -s -X POST localhost:8000/pricing/price -H 'Content-Type: application/json' -d '{
  "surface_id":"tech","version":1,"option_type":"vanilla_call",
  "expiry":2.0,"strike":250,"allow_extrapolation":true}'

# 批量：整批绑同一版本，逐条成功/失败
curl -s -X POST localhost:8000/pricing/price/batch -H 'Content-Type: application/json' -d '{
  "surface_id":"default","version":1,"items":[
    {"option_type":"vanilla_call","expiry":1,"strike":100},
    {"option_type":"american_put","expiry":1,"strike":100,"resolution":252},
    {"option_type":"vanilla_call","expiry":99,"strike":100}]}'
```

## 6. 验证方法

```bash
uv run pytest                       # 全部用例（78 个）
uv run pytest tests/test_pricing_benchmark.py -v   # 解析基准对照
uv run pytest tests/test_resolution.py -v          # 固定分辨率/上下边界/确定性
uv run pytest tests/test_resolution_types.py -v    # 本次修复：布尔/浮点/字符串
                                                   # 在同步/异步/批量三入口被拒
uv run pytest tests/test_extrapolation.py -v       # 外插授权/审计/历史版本
uv run pytest tests/test_batch.py -v               # 批量：部分失败/版本一致/慢条隔离
uv run pytest tests/test_validation.py -v          # 非法/越界/不收敛
uv run pytest tests/test_concurrency.py -v         # 并发发布/版本回放
uv run pytest tests/test_api.py -v                 # HTTP 契约/取消/外插/批量
```

本次分辨率类型收紧的本地复现/验证：

```bash
# 1) 模型层直接复现旧缺陷已被修复（全部应打印 REJECTED）
uv run python -c "
from pydantic import ValidationError
from option_engine.models import PricingRequest
for v in [True, False, 1.0, 1.5, '100', '1.0']:
    try:
        PricingRequest(option_type='american_put', expiry=1.0,
                       strike=100, resolution=v)
        print(repr(v), 'ACCEPTED (BUG)')
    except ValidationError:
        print(repr(v), 'REJECTED')"

# 2) 三入口行为对照（同步/异步为 422；批量为该条 failed、整批 200）
uv run uvicorn main:app --port 8000
# 然后使用上方“使用示例”中的 true / 0 / 批量三条目 curl 命令验证。

# 3) 专门的回归测试（含边界 1 与 2000 仍按精确步数计算、同版本逐位相同）
uv run pytest tests/test_resolution_types.py -v
```

测试日志（stdout 与 caplog 均有）对每个用例打印：

```
CASE=version_replay | INPUTS={'versions': [1, 2, 3]} | SURFACE_VERSION=[1,2,3]
  | CRITERION=each version reproduces its own price; sync and async paths identical...
```

即**输入参数、所用曲面版本、判定依据**三要素，便于审计与复现。

## 7. 目录结构

```
option_engine/
  errors.py      # 可区分失败原因的异常层级
  resolution.py  # resolution 唯一校验入口（允许类型/区间/边界；bool/float/str 拒绝）
  models.py     # Pydantic 请求/响应模型（含分辨率严格类型、外插范围、批量模型）
  surface.py    # 不可变快照、双线性插值、外插授权与审计、版本化存储（原子发布）
  pricing.py    # BSM/数字闭式解、CRR 美式（自动收敛/固定分辨率）、隐含波动率
  service.py    # 版本绑定 + 规范化结果缓存（跨路径一致性）
  batch.py      # 批量定价：单版本绑定、逐条并发、单条失败隔离
  jobs.py       # 线程池异步任务、状态机、协作取消
  data.py       # 本地 JSON 曲面载入
  api.py        # FastAPI 路由与错误映射
data/surfaces/  # 本地曲面数据（default v1/v2 禁外插、tech v1 声明外插范围）
tests/          # 基准/分辨率/分辨率类型/外插/批量/非法参数/并发/HTTP 八组测试
```
