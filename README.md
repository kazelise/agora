# Agora

多 Agent 和人待在同一间房里的聊天后端。新消息会叫醒房间里的 Agent；每个 Agent 串行跑 turn，突发叫醒合并成一轮，避免 N 条消息打出 N 次推理。项目要解决的是多 Agent 协作里的两类失败——抢答碰撞和脑判失误——并把「时间窗上的竞态」交给代码、「语义上的对错」交给模型。目前完成到 Phase 1：房间 / 消息 / WebSocket / Redis 叫醒调度。没有前端，演示靠 CLI、日志和测试。

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

Inspired by Cumora (github.com/yetone/cumora); independently designed and implemented from scratch.

设计说明见 [docs/design.md](docs/design.md)。

## 怎么跑

本机若 5432 已被占用，compose 把 Postgres 映到 **5433**（容器内仍是 5432）。Redis 用 6379。

```bash
cd agora
docker compose up -d
uv sync
export AGORA_DATABASE_URL=postgresql://agora:agora@127.0.0.1:5433/agora
export AGORA_REDIS_URL=redis://127.0.0.1:6379/0
uv run uvicorn server.main:app --reload --port 8000
```

另开一个终端跑 Phase 1 demo（进程内拉起应用，不依赖上面的 uvicorn，但仍要 Postgres + Redis）：

```bash
uv run python scripts/demo_phase1.py
```

## 测试

```bash
docker compose up -d
uv sync
uv run pytest
```

`test_coalesce` 不需要外部服务。`test_seq` 需要 Postgres；`test_wake` 需要 Postgres + Redis。conftest 在连不上时会尝试 `docker compose up`；若 Docker 也不可用，集成测试会被 skip。
