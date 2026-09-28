# 期权定价与波动率曲面服务（本地独立运行）

一套可完全离线运行的期权定价服务：波动率与行情全部来自本地曲面数据，
同一市场状态在不同调用路径（同步 HTTP / 异步任务 / 批量 / 直接调用）下结论一致、
可复现；非法输入、曲面越界、数值不收敛一律**拒绝并返回可区分的失败原因**，
绝不返回近似值或默认值。支持**指定固定离散分辨率**的确定性美式定价、
**按曲面版本声明且需调用方显式授权的有界外插（用必审计）**，以及
**整批绑同一版本、逐条并发、部分失败互不影响的批量定价**。

- 纯标准库实现数值部分（`math.erf` 正态分布、CRR 二叉树、二分法），无需 numpy/scipy
- FastAPI + Pydantic，`uv run uvicorn main:app` 即可启动
- Python >= 3.14

## 1. 定价模型

| 期权类型 | 方法 | 说明 |
| --- | --- | --- |
| `vanilla_call` / `vanilla_put` | Black–Scholes–Merton 闭式解 | 连续股息率 q |
| `digital_call` / `digital_put` | BSM 现金或无（cash-or-nothing）闭式解 | 收益为固定 `payout` |
| `american_put` | Cox–Ross–Rubinstein 二叉树 | **两种精度模式**（见下）：自动收敛 或 指定固定分辨率；到期前可提前行权 |
| `target_price`（欧式香草） | 隐含波动率二分求解 | 反解市场价格对应的 IV |

### 1.1 离散精度控制（回测复现）

美式期权提供两种**可区分**的定价模式，响应里以 `resolution_mode` /
`resolution_used` 明确标识：

| 模式 | 请求方式 | 行为 | 响应字段 |
| --- | --- | --- | --- |
| 自动收敛（默认） | 不传 `resolution`，用 `steps`/`convergence_tol`/`max_steps` | 步数倍增直到相对变化 < 容差；预算用尽则拒绝（`solver_non_convergence`），绝不返回未收敛近似值 | `resolution_mode="converged"`，`resolution_used`=实际收敛步数 |
| 固定分辨率 | `resolution=N`（**仅** `american_put`） | **恰好**跑 N 步 CRR 树，不做收敛循环、不偷偷替换成任何其它步数 | `resolution_mode="fixed"`，`resolution_used=N` |

- 允许范围：`resolution ∈ [1, 10000]`（`MIN_RESOLUTION`/`MAX_RESOLUTION`，
  在 `option_engine/models.py` 中集中定义）。**边界值 1 与 10000 都能正常算出
  价格**；范围外（0、负数、>10000）、布尔值、小数一律在模型层拒绝
  （`StrictInt`，HTTP 422 `invalid_parameter`），引擎层对绕过模型的请求有同样的
  兜底校验；不会把任何请求悄悄夹到别的分辨率。
- 确定性：同一请求绑定同一曲面版本反复跑结果完全一致（纯函数 + 按规范化请求
  哈希的结果缓存）。固定模式不设 `converged` 标记：`detail={"steps_used": N,
  "converged": false}`，与自动收敛模式一眼可分。
- `resolution` 与 `steps/convergence_tol/max_steps` 互斥（同时给非默认收敛参数
  会被拒绝），避免“到底按哪个跑”的歧义。
- 典型用途：复现交易对手“固定 250 步”给出的美式价格——直接传
  `"resolution": 250`，拿到的就是标准 250 步 CRR 树价格。

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
- **外插采用“发布时授权 + 请求时显式申请”双重控制，默认仍然越界即拒绝**：
  - 曲面发布时用 `extrapolation_limits` 声明**可外插的矩形范围**
    （`min_expiry/max_expiry/min_strike/max_strike`，必须包含整个网格支撑，
    否则发布被拒）；不声明（缺省）则该版本**完全不可外插**；
  - 调用方在请求里显式带 `"allow_extrapolation": true` 才允许在该版本声明的
    范围内取外插值；不带则越界点照常抛 `SurfaceOutOfRangeError`；
  - 对没有声明范围的版本显式申请外插 →
    `ExtrapolationNotAuthorizedError`（`extrapolation_not_authorized`，
    HTTP 422），与“普通越界拒绝”明确区分；
  - 即便已授权，查询点超出该版本声明范围仍抛 `SurfaceOutOfRangeError`
    （不会夹到边界、不会换近似值）；
  - 旧写法 `allow_extrapolation`（随曲面写死的布尔开关）已移除，改为
    `extrapolation_limits`；
- **外插审计**：只要实际用到了外插，响应必带 `"extrapolated": true` 和
  `extrapolation` 报告（网格内点恒为 `false`/`null`）。报告逐轴给出
  外插方向、网格边界、声明边界、外插绝对距离以及“已用掉声明余量的比例”
  （`headroom_fraction_used`），例如到期日 2.0、网格上限 1.0、声明上限 3.0
  时该比例为 0.5。绝不会悄悄返回一个外插价格而不标注；
- **历史版本回放的一致性**：授权与否、可外插范围都冻结在版本快照里。
  v1 没声明范围则永远不可外插；之后发布的 v2 声明了范围也不改变 v1 的行为，
  绑定旧版本重放时遵守的就是该版本当时的声明；
- 每次 `POST /surfaces/{id}/versions` 生成一个**单调递增新版本号**，
  旧版本永不修改、永不删除，可永久回放；
- 定价请求可用 `version` 绑定历史版本；不传则在处理开始时锁定当时最新版本，
  并在响应/任务状态中返回 `surface_version`（任务失败时该版本号同样保留）；
- 启动时自动载入 `data/surfaces/*.json`（命名 `<id>_v<n>.json`，
  同 id 按文件名顺序发布为 v1、v2…）。当前内置：
  - `default` v1（S=100,r=5%）、v2（S=102,r=4.5%,q=1%）——均不声明外插范围，
    用于版本回放对照；
  - `tech` v1（S=200，声明外插范围 到期日 [0.1, 3.0]、行权价 [150, 300]）。

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

### 3.1 批量定价（整本书绑同一版本）

`POST /pricing/batch` 一次提交一整批请求（如一天的组合）：

- **整批绑同一个曲面版本**：版本在处理任何条目之前只解析一次并钉住
  （不传 `version` 则在提交时钉住当时最新版）；处理期间即使有新版本发布，
  批内条目也不会被拆到不同版本，响应顶层与每条结果都带该版本号；
- 条目不允许自带 `surface_id`/`version`（`BatchItem` 为 `extra: forbid`，
  混入即整批 422）；批量大小 1～1000（`MAX_BATCH_ITEMS`），条目 `id` 可选
  但在批内必须唯一；
- **逐条独立、并发执行**：条目在独立线程池（最多 8 线程）里各走与单次
  `POST /pricing/price` **完全相同**的 `PricingService.price` 路径
  （同一校验、同一结果缓存）。因此同一版本下，批量里每条的结果与单独调用
  严格一致；
- **部分失败语义**：HTTP 请求本身恒为 200（除非批量形状非法），每条单独给出
  结果或错误：`items[i] = {"index", "id", "result" | "error"}`，二者恰有一个。
  某条越界、参数非法或不收敛只落在它自己的 `error.code/message` 上，
  不影响也不拖慢其他条目（慢条目只占它自己的工作线程）；
- 条目的固定分辨率、外插授权等控制与单次请求完全相同，成功条目同样带
  `resolution_mode/resolution_used` 与 `extrapolated/extrapolation` 审计字段。

## 4. HTTP 接口

| 方法 & 路径 | 作用 |
| --- | --- |
| `GET /health` | 健康检查 |
| `POST /pricing/price` | 同步定价（支持 `resolution` 固定分辨率、`allow_extrapolation`） |
| `POST /pricing/batch` | 批量定价（整批绑一个版本，逐条并发、部分失败隔离） |
| `POST /surfaces/{id}/versions` | 发布新曲面版本（可带 `extrapolation_limits` 声明外插范围） |
| `GET /surfaces/{id}/versions` | 列出全部版本 |
| `GET /surfaces/{id}/versions/{v}` | 查询指定版本元数据（含声明的外插范围） |
| `GET /surfaces/{id}/versions/{v}/vol?expiry=&strike=&allow_extrapolation=` | 查询波动率（含外插审计字段） |
| `POST /jobs` | 提交异步定价任务（202，返回 job_id） |
| `GET /jobs/{id}` | 查询任务状态/结果/失败原因 |
| `POST /jobs/{id}/cancel` | 取消任务 |

错误响应统一格式：

```json
{"error": "surface_out_of_range", "message": "expiry 99.0 outside ...", "detail": {}}
```

| error 码 | 含义 | HTTP |
| --- | --- | --- |
| `invalid_parameter` | 参数非法 / pydantic 校验失败 / IV 无可行解 / 分辨率越界或类型不对 | 422 |
| `surface_out_of_range` | 查询点在网格外：未申请外插，或超出该版本声明的外插范围 | 422 |
| `extrapolation_not_authorized` | 显式申请外插，但所绑版本未声明任何外插范围 | 422 |
| `solver_non_convergence` | 数值求解未收敛 | 422 |
| `surface_not_found` / `job_not_found` | 曲面/版本/任务不存在 | 404 |
| `job_cancelled` | 任务被取消 | 任务体内 |
| `internal_error` | 未预期错误（不会击垮工作线程/批量线程） | 500 |

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

# 固定 250 步复现美式价格（回测复现）；响应带 resolution_mode/used
curl -s -X POST localhost:8000/pricing/price -H 'Content-Type: application/json' -d '{
  "surface_id":"default","option_type":"american_put","expiry":1,"strike":100,
  "resolution":250}'

# 发布一个声明外插范围的曲面版本
curl -s -X POST localhost:8000/surfaces/demo/versions -H 'Content-Type: application/json' -d '{
  "expiries":[0.25,1.0],"strikes":[90,100,110],
  "vols":[[0.22,0.20,0.19],[0.23,0.205,0.195]],"spot":100,"rate":0.05,
  "extrapolation_limits":{"min_expiry":0.1,"max_expiry":2.0,
                          "min_strike":80,"max_strike":130}}'

# 在声明范围内显式外插 -> 200，响应 extrapolated=true 且带逐轴审计
curl -s -X POST localhost:8000/pricing/price -H 'Content-Type: application/json' -d '{
  "surface_id":"demo","option_type":"vanilla_call","expiry":1.5,"strike":125,
  "allow_extrapolation":true}'

# 同点不申请 -> 422 surface_out_of_range；超范围申请 -> 同样 422
curl -s -X POST localhost:8000/pricing/price -H 'Content-Type: application/json' -d '{
  "surface_id":"demo","option_type":"vanilla_call","expiry":1.5,"strike":125}'

# 批量：整批绑一个版本，逐条结果/错误
curl -s -X POST localhost:8000/pricing/batch -H 'Content-Type: application/json' -d '{
  "surface_id":"default",
  "items":[
    {"id":"ok","option_type":"vanilla_call","expiry":1,"strike":100},
    {"id":"oob","option_type":"vanilla_call","expiry":99,"strike":100},
    {"id":"am","option_type":"american_put","expiry":1,"strike":100,"resolution":64}
  ]}'

# 异步美式期权
curl -s -X POST localhost:8000/jobs -H 'Content-Type: application/json' -d '{
  "option_type":"american_put","expiry":1,"strike":100,
  "steps":500,"max_steps":4000,"convergence_tol":1e-4}'
```

## 5. 验证方法

```bash
uv run pytest                       # 全部用例
uv run pytest tests/test_pricing_benchmark.py -v   # 解析基准对照
uv run pytest tests/test_validation.py -v          # 非法/越界/不收敛/外插授权
uv run pytest tests/test_resolution_batch.py -v    # 固定分辨率/外插审计/批量
uv run pytest tests/test_concurrency.py -v         # 并发发布/版本回放
uv run pytest tests/test_api.py -v                 # HTTP 契约/取消
```

`tests/test_resolution_batch.py` 重点覆盖：

- **固定分辨率**：1/2/3/17/100 步结果与直接调用 `american_put_crr(n)` 逐位相等，
  响应模式为 `fixed` 且 `resolution_used==n`；同版本重复请求严格一致（含缓存）；
- **边界行为**：允许范围 `[1, 10000]` 的**两个端点都真实算出价格**
  （10000 步结果同样与直接求解相等），0/-1/10001/1e9 与布尔、小数全部拒绝，
  证明“没到上限就误判不可算”和“静默换成近似分辨率”都不会发生；
- **模式区分**：fixed 与 converged 的 `resolution_mode/steps/converged` 一眼可分；
  `resolution` 仅美式可用且与收敛参数互斥；
- **外插**：显式授权且在声明范围内 → 给价并逐轴审计（方向、边界、余量比例）；
  不授权越界 → `surface_out_of_range`；对未声明范围的版本申请 →
  `extrapolation_not_authorized`；超出声明范围即使授权仍
  `surface_out_of_range`；同一点在批量内按条目各自授权/拒绝；
- **历史版本**：v1 无范围永远拒绝、v2 声明范围可服务且范围随版本冻结；
- **批量**：部分成功/部分失败共存且按下标与 id 对齐；批量每条结果与单独调用
  相等；并发发布压测下每批只钉一个版本；慢条目（最终不收敛）不阻塞同批便宜
  条目；批量 HTTP 契约（空批/超大批/重复 id/私带 version → 422）。

测试日志（stdout 与 caplog 均有）对每个用例打印：

```
CASE=version_replay | INPUTS={'versions': [1, 2, 3]} | SURFACE_VERSION=[1,2,3]
  | CRITERION=each version reproduces its own price; sync and async paths identical...
```

即**输入参数、所用曲面版本、判定依据**三要素，便于审计与复现。

## 6. 目录结构

```
option_engine/
  errors.py     # 可区分失败原因的异常层级（含 extrapolation_not_authorized）
  models.py     # Pydantic 模型：精度模式/分辨率边界、外插范围、批量请求-响应
  surface.py    # 不可变快照、双线性插值、按版本声明且有界的外插、原子发布
  pricing.py    # BSM/数字闭式解、CRR 美式（固定分辨率/自动收敛）、隐含波动率
  service.py    # 版本绑定 + 规范化结果缓存 + 批量并发定价（部分失败隔离）
  jobs.py       # 线程池异步任务、状态机、协作取消
  data.py       # 本地 JSON 曲面载入
  api.py        # FastAPI 路由与错误映射（含 /pricing/batch）
data/surfaces/  # 本地曲面数据（default v1/v2 无外插、tech v1 声明外插范围）
tests/          # 基准/非法参数/精度与外插与批量/并发/HTTP 五组测试
```
