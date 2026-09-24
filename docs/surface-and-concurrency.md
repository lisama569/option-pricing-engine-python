# 曲面版本管理与并发处理规则

## 1. 数据来源与启动加载

- 所有输入均来自本地：曲面 JSON 文件位于 `data/surfaces/`
  （可用环境变量 `OPTION_ENGINE_DATA_DIR` 覆盖目录；线程池大小由
  `OPTION_ENGINE_MAX_WORKERS` 控制，默认 4）。
- 启动时按**文件名排序**逐个原子发布目录内全部 `.json`，
  因此同一份数据目录在任何机器上产生相同的版本号序列（v1、v2……）。
- 数据目录缺失/为空不阻断启动：服务以空仓库启动，随后可用
  `POST /api/v1/surfaces` 发布；但加载到坏文件会整体中止并报错，
  绝不静默跳过坏数据。

## 2. 曲面结构与校验（`SURFACE_DATA_INVALID`）

- 到期日、行权价严格递增且为正，各自至少 2 个点；
- `vols` 为 `len(maturities) × len(strikes)` 矩形，全部波动率严格为正
  （NaN/0/负数拒绝）；无风险利率、股息率非负。

## 3. 插值规则（`SURFACE_OUT_OF_RANGE`）

- 盒内采用**分片双线性插值**：先在相邻两个到期日切片上对行权价线性插值，
  再对到期日线性插值。映射确定唯一，无随机数、无优化器参与。
- 查询点必须严格落在网格包围盒内（含边界，带 1e-10 相对容差）。
- **禁止外插**：到期日或行权价落在网格外一律拒绝，绝不返回邻近值、
  端点值或任何默认/近似值。

## 4. 版本化与历史回放

- 每次发布先校验、构造完整快照，再在 `RLock` 临界区内做**一次引用追加**，
  版本号单调递增；历史版本永不修改、永不删除。
- `SurfaceStore.get()` 返回快照的**深拷贝**，调用方的原地修改无法污染仓库。
- 请求可带 `surface_version` 绑定历史版本（不带则使用最新）。
  因为（请求参数，曲面版本）→（波动率，无风险利率，股息率）→ 价格是
  纯函数链，相同输入 + 相同版本在任何调用路径（同步 `/price`、异步任务、
  IV 反解）下得到**逐位一致**、可复现的结论。

## 5. 并发安全

- **曲面更新期间的读取**：发布是“先构造完整新快照，再整体替换引用”，
  读者要么看到旧版本、要么看到新版本，不可能看到半更新矩阵或新旧混杂；
  快照发布后不可变，读者拿到引用即可在锁外使用。压测见
  `tests/test_surface.py::test_concurrent_publish_never_shows_partial_state`
  与 `tests/test_api.py::test_publish_and_concurrent_reads_never_mixed`。
- **慢任务不阻塞**：定价在独立 `ThreadPoolExecutor` 中执行，HTTP 事件循环
  与其他请求不被大步数二叉树阻塞（见 `test_slow_job_does_not_block_fast_job`）。
- **任务状态机**：`pending → running → succeeded/failed`，或
  `pending → cancelled`。只有仍在排队的任务可取消（运行中的线程无法安全
  强杀，会跑完）；重复取消返回 `JOB_ALREADY_DONE`。
- **失败隔离**：任何任务异常都被捕获并落成 `failed + 结构化错误`，
  不影响线程池、其他任务与整体服务（见 `test_failed_job_is_structured_and_service_continues`）。

## 6. 失败原因对照表（可区分、可判定）

| error_code | HTTP | 含义 |
| --- | --- | --- |
| `INVALID_MODEL_PARAMS` | 422 | 标的价/行权价非正、到期为负、数字期权缺 cash、美式缺/非法 tree_steps、IV 用于非欧式香草等 |
| `SURFACE_OUT_OF_RANGE` | 422 | 到期日/行权价落在曲面网格外，拒绝外插 |
| `SOLVER_DID_NOT_CONVERGE` | 422 | 目标价超出可解区间或 100 次迭代未收敛（输入本身合法但无解） |
| `SURFACE_DATA_INVALID` | 422 | 发布的曲面数据结构/取值不合法或 JSON 损坏 |
| `VERSION_NOT_FOUND` | 404 | 绑定了不存在的曲面版本，或仓库尚为空 |
| `JOB_NOT_FOUND` | 404 | 查询/取消不存在的任务 |
| `JOB_ALREADY_DONE` | 409 | 取消已 running/succeeded/failed/cancelled 的任务 |

注：目标价 ≤ 0 这类“输入本身非法”在服务层归为 `INVALID_MODEL_PARAMS`；
“输入合法但数值上不存在解”才归为 `SOLVER_DID_NOT_CONVERGE`。
