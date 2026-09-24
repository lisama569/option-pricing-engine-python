# 期权定价与波动率曲面服务（本地独立运行）

纯 Python 标准库实现数值核心（正态分布、BSM 闭式解、CRR 二叉树、Brent 反解），
FastAPI 提供 HTTP 接口。所有曲面与行情输入均来自本地 JSON 文件，
同一市场状态（曲面版本）在同步定价、异步任务等不同调用路径下结论**逐位一致、可复现**。

## 快速开始

```bash
# 依赖仅 fastapi[standard]（测试需 pytest），无需 numpy/scipy
uv sync                                  # 或直接使用系统 Python（>=3.14）
uvicorn main:app --host 127.0.0.1 --port 8000
# 或：python3 main.py
```

启动时自动按文件名排序发布 `data/surfaces/` 下的曲面（当前附带 v1/v2 两个版本）。
交互式文档：<http://127.0.0.1:8000/docs>。

## API 一览（前缀 `/api/v1`）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康状态与当前最新曲面版本 |
| POST | `/surfaces` | 原子发布一个新曲面版本（整体替换引用，无中间态） |
| GET | `/surfaces` | 列出全部版本 |
| GET | `/surfaces/{version}` | 查询指定版本元信息 |
| POST | `/pricing/price` | 同步定价（请求体可带 `surface_version` 绑定历史版本） |
| POST | `/pricing/implied-vol` | 欧式香草期权隐含波动率反解 |
| POST | `/jobs` | 提交异步定价任务（线程池执行，202 返回 job_id） |
| GET | `/jobs` / `/jobs/{id}` | 列出/查询任务状态与结果 |
| POST | `/jobs/{id}/cancel` | 取消仍在排队（pending）的任务 |

定价请求示例：

```bash
curl -s http://127.0.0.1:8000/api/v1/pricing/price \
  -H 'Content-Type: application/json' \
  -d '{"option_type":"european_call","spot":100,"strike":100,
       "maturity":0.5,"surface_version":1}'
```

响应含 `price`、`greeks`、`implied_vol`（曲面内插得到的波动率）、
`surface_version`、`surface_label`、`model`。

支持的 `option_type`：`european_call/put`、`cash_or_nothing_call/put`
（需带正的 `cash`）、`american_call/put`（需带 `tree_steps`，1~20000）。

## 曲面数据格式（本地 JSON）

```json
{
  "label": "SURF-2026Q3",
  "as_of": "2026-09-24",
  "maturities": [0.25, 0.5, 1.0],
  "strikes": [90.0, 95.0, 100.0, 105.0, 110.0],
  "vols": [[0.26, 0.245, 0.24, 0.245, 0.26], "...每个到期日一行..."],
  "risk_free_rate": 0.035,
  "dividend_yield": 0.015
}
```

波动率取点为**严格分片双线性内插，禁止外插**；越界、非法参数、求解不收敛
均以稳定 `error_code` 拒绝（422/404/409），不返回近似或默认值。完整对照表见
[`docs/surface-and-concurrency.md`](docs/surface-and-concurrency.md)。

## 验证方法

```bash
python3 -m pytest -s          # 52 个用例；-s 可看到每条用例打印的
                              # 输入参数、曲面版本与判定依据
```

测试覆盖：

1. **解析基准对照**：正态 CDF 查表值、Hull 教材 ATM 看涨/看跌价格
   （10.45058357 / 5.57352602）、买卖权平价、数字期权恒等式、
   Greeks 对中心差分、CRR 收敛性与提前行权溢价、IV 反解往返；
2. **非法与越界参数**：三类业务失败原因（`INVALID_MODEL_PARAMS` /
   `SURFACE_OUT_OF_RANGE` / `VERSION_NOT_FOUND`）及 IV 的
   `SOLVER_DID_NOT_CONVERGE` 互相可区分；
3. **并发曲面更新与历史版本回放**：多线程发布/读取下无部分可见状态、
   绑定 v1 的定价在发布 v2..v7 后逐位不变；
4. **并发任务**：相同输入+版本结果唯一、慢任务不阻塞快任务、
   失败任务结构化落盘且服务继续、pending 任务取消。

模型公式与数值细节：[`docs/pricing-models.md`](docs/pricing-models.md)；
版本与并发规则：[`docs/surface-and-concurrency.md`](docs/surface-and-concurrency.md)。
