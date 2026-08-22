# Agora

多 Agent 和人待在同一间房里的聊天后端。新消息会叫醒房间里的 Agent；每个 Agent 串行跑 turn，突发叫醒合并成一轮，避免 N 条消息打出 N 次推理。项目要解决的是多 Agent 协作里的两类失败——抢答碰撞和脑判失误——并把「时间窗上的竞态」交给代码、「语义上的对错」交给模型。目前完成到 Phase 5：房间 / 消息 / WebSocket / Redis 叫醒调度，一张 LangGraph（小模型 triage、大模型 `reply`/`claim`、代码节点 freshness HOLD、提交时事务内新鲜度校验、按 seq 锚定的 `task_key`、`llm_calls` 账本），计数游戏 / one-of-us 两条真模型协调测试，BYOA，可选的云端 K8s Job 宿主，GitHub OAuth + JWT 准入（未配置 `AGORA_GITHUB_CLIENT_ID` 时仍是匿名 curl），以及 Redis 上的多 worker presence / 广播 / 车道锁。没有前端，演示靠 CLI、日志和测试。

```mermaid
flowchart LR
  subgraph clients [接入]
    CLI[agora CLI / curl]
    OAuth[GitHub OAuth + JWT]
  end
  subgraph server [FastAPI 服务端]
    API[REST + WebSocket]
    Scheduler[唤醒调度器<br/>per-agent 串行 + 突发合并]
    Ledger[(llm_calls 成本账本)]
  end
  subgraph brain [同一张 LangGraph]
    Triage[triage 节点<br/>小模型: me/each/one-of-us]
    Loop[工具循环节点<br/>大模型: reply/claim]
    Fresh[freshness 节点<br/>过期则 interrupt-HOLD]
  end
  PG[(Postgres<br/>消息/认领/checkpoint)]
  RD[(Redis<br/>pub/sub + seen cursor)]
  subgraph hosts [两种 Computer]
    K8sJob[云端: K8s Job]
    Daemon[BYOA daemon<br/>用户自己的 key]
  end
  CLI --> API
  OAuth --> API
  API --> PG
  API --> RD
  RD --> Scheduler
  Scheduler --> K8sJob
  Scheduler --> Daemon
  K8sJob --> brain
  Daemon --> brain
  brain --> API
  brain --> Ledger
```

人的准入和宿主的钥匙是两套凭据：GitHub OAuth 换成 JWT，管房间 / Computer / 房间 WebSocket；Computer token 和 cluster token 只管 `/runtime/*` 和 Computer WebSocket。JWT 打不进 runtime，配对 token 当不了登录用户。未配置 GitHub client id 时准入关闭，本机 demo 和现有测试不用改。云端宿主默认仍是进程内 `DirectWorld`；打开 `AGORA_K8S_ENABLED` 后走 Job。

Inspired by Cumora (github.com/yetone/cumora); independently designed and implemented from scratch.

设计说明见 [docs/design.md](docs/design.md)。

## 怎么跑

本机若 5432 已被占用，compose 把 Postgres 映到 **5433**（容器内仍是 5432）。Redis 用 6379。

```bash
cd agora
docker compose up -d
uv sync
# 变量说明见仓库根目录 .env.example
export AGORA_DATABASE_URL=postgresql://agora:agora@127.0.0.1:5433/agora
export AGORA_REDIS_URL=redis://127.0.0.1:6379/0
uv run uvicorn server.main:app --reload --port 8000
```

另开一个终端跑 demo（进程内拉起应用，不依赖上面的 uvicorn，但仍要 Postgres + Redis）：

```bash
uv run python scripts/demo_phase1.py          # 叫醒 / 合并，走 turn 桩

# 真模型 demo：本地 OpenAI 兼容中继，不需要真实 key
export OPENAI_API_KEY=relay-no-key
export OPENAI_BASE_URL=http://192.168.1.100:8317/v1
export OPENAI_API_BASE=$OPENAI_BASE_URL
export AGORA_SMALL_MODEL=gpt-5.6-luna          # triage，默认即此
export AGORA_BIG_MODEL=gpt-5.6-terra           # 工具循环，默认即此
uv run python scripts/demo_phase2.py           # one-of-us 介绍房间
uv run python scripts/demo_byoa.py             # 云端 + 本地 daemon，然后把 daemon 杀掉
```

## BYOA 快速开始

Computer 是宿主：`computer_id` 为空的 Agent 走进程内云端车道；有值的走那台机器上的 daemon。daemon 用自己的 `OPENAI_API_KEY` 跑同一张图，只通过 `/runtime/*` 读写世界，服务端看不到 key。

另开一个终端（服务已经在 8000 端口）：

```bash
# 1. 配对一台 Computer（token 只在这一次响应里出现）
curl -s http://127.0.0.1:8000/computers \
  -H 'content-type: application/json' \
  -d '{"name":"my-laptop"}'
# → {"id":"...","name":"my-laptop","token":"..."}

# 2. 建房、加人、把 Agent 挂到这台 Computer（把 COMPUTER_ID 换成上一步的 id）
#    POST /rooms  →  POST /rooms/{id}/participants
#    人：{"kind":"human","name":"Ada"}
#    Agent：{"kind":"agent","name":"Jules","computer_id":"<COMPUTER_ID>"}

# 3. 跑 daemon（环境变量里是 *daemon 自己的* key，不是服务端的）
export AGORA_SERVER_URL=http://127.0.0.1:8000
export AGORA_COMPUTER_ID=<id>
export AGORA_COMPUTER_TOKEN=<token>
export OPENAI_API_KEY=relay-no-key
export OPENAI_BASE_URL=http://192.168.1.100:8317/v1
export OPENAI_API_BASE=$OPENAI_BASE_URL
uv run python -m daemon

# 4. 再往房间 POST 一条人的消息。daemon 日志就是演示：
#    wake → triage → claim/hold/reply。Computer 断线则 Agent 显示 sleeping。
```

一条命令看完整故事（进程内拉起服务、配对、拉起 daemon 子进程、人提问、再杀掉 daemon）：

```bash
uv run python scripts/demo_byoa.py
```

## 测试

```bash
docker compose up -d
uv sync
uv run pytest -m "not llm"    # mock 模型，不打中继
uv run pytest -m llm          # 真中继：计数游戏 + one-of-us（需要上面的 OPENAI_*）
```

开 GitHub 准入：设 `AGORA_GITHUB_CLIENT_ID` / `AGORA_GITHUB_CLIENT_SECRET` / `AGORA_JWT_SECRET`，浏览器走 `GET /auth/github`。回来的 JWT 放 `Authorization: Bearer`，房间 WebSocket 用 `?access_token=`。房间和 Computer 归创建者所有。

`test_coalesce`、Job manifest / launcher、JWT 和模型策略测试不需要外部服务。`test_seq` 需要 Postgres；`test_wake` / `test_brain` / `test_seen` / `test_k8s` / `test_oauth` / `test_cluster` 的 HTTP 用例需要 Postgres + Redis。`-m "not llm"` 全部 mock 模型。`-m llm` 打真实中继，未设置 `OPENAI_BASE_URL` 或中继不可达时会 skip。conftest 在连不上时会尝试 `docker compose up`；若 Docker 也不可用，集成测试会被 skip。

开 K8s Job 宿主见 [k8s/README.md](k8s/README.md)。
