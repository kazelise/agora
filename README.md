# Agora

多 Agent 和人待在同一间房里的聊天后端。新消息会叫醒房间里的 Agent；每个 Agent 串行跑 turn，突发叫醒合并成一轮，避免 N 条消息打出 N 次推理。项目要解决的是多 Agent 协作里的两类失败——抢答碰撞和脑判失误——并把「时间窗上的竞态」交给代码、「语义上的对错」交给模型。目前完成到 Phase 6：房间 / 消息 / WebSocket / Redis 叫醒调度，一张 LangGraph（小模型 triage、大模型 `reply`/`claim`、代码节点 freshness HOLD、提交时事务内新鲜度校验与逐字复读拦截、按 seq 锚定的 `task_key`、`llm_calls` 账本、把 `send_anyway` 当确认而非跳过的 hold token、房间级的 agent-only 循环上限），计数游戏 / one-of-us 两条真模型协调测试，BYOA——同一张图跑在用户自己的 Computer 上，服务端不持有用户的模型 key——可选的云端 K8s Job 宿主（未开 k8s 时回退进程内），以及 `GET /rooms/{id}/digest` 把房间沉淀为 Markdown brief（transcript / active claims / 模型花费，纯格式化零模型调用）。没有前端，演示靠 CLI、日志和测试。

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

图中 OAuth 为规划中的准入层，尚未实现。云端宿主默认仍是进程内 `DirectWorld`；打开 `AGORA_K8S_ENABLED` 后，同一条 per-agent 车道会为 `computer_id` 为空的 Agent 创建一个 Job（`python -m brain.job`，cluster token + `HttpWorld`）。BYOA daemon 不变。

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

同一台 Computer 上的多个 Agent 同时被叫醒时，daemon 内置的 AdaptivePacer
会把模型调用按最小间隔错峰起步（默认 0.5s）；收到 429 时间隔指数加倍
（上限 8s），连续干净调用再折半回落。可用 `AGORA_PACER_BASE_S` /
`AGORA_PACER_MAX_S` 调整。

pacer 错开的是起步时刻，不限制同时在飞的调用数——同一进程还有
ConcurrencyLimiter 把在飞的模型调用封顶（默认 6，两层模型共用一个预算；
配额紧张可降到 2–4）。可用 `AGORA_MAX_CONCURRENT` / `--max-concurrent` 调整。

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

`test_coalesce`、Job manifest / launcher 和模型策略测试不需要外部服务。`test_seq` 需要 Postgres；`test_wake` / `test_brain` / `test_seen` / `test_k8s` 的 runtime 用例 / `test_hardening`（hold token、verbatim-dup、循环上限、digest）需要 Postgres + Redis。`-m "not llm"` 全部 mock 模型。`-m llm` 打真实中继，未设置 `OPENAI_BASE_URL` 或中继不可达时会 skip。conftest 在连不上时会尝试 `docker compose up`；若 Docker 也不可用，集成测试会被 skip。

## 房间 digest

讨论的沉淀物，一行命令拉取（`room-<id>.md`，可直接贴进 issue 或笔记）：

```bash
curl -s http://127.0.0.1:8000/rooms/<ROOM_ID>/digest -o room.md
```

内容：transcript 表格、active claims（即 action items）、`llm_calls` 按 purpose × model 汇总的花费。纯格式化，零模型调用。

## 沉默房间的主动唤醒（stall sweep）

turn 都是反应式的——叫醒只在新消息落地时发生。但「有人欠话」的房间一旦安静（claim 赢家认栽释放、提问没人接），就没有任何机制再叫醒人。服务端内置的 `StallSweeper` 周期扫描：房间最新消息安静超过 20s（`AGORA_STALL_MIN_S`）、且至少一个非作者 agent 已读过它时，以「最后发言者 = 名义发信人」走 `dispatch` 通道主动唤醒房里其余 agent——BYOA agent 收到 computer websocket wake、云 agent 收到 K8s Job，与真实消息同路径，离线宿主照旧不排队。nudge 之后房间依然沉默则记一次 decline，`AGORA_STALL_MAX_NUDGES`（默认 3）次后停手，直到任何新消息落地重置预算。

被唤醒的 agent 已读全部消息、inbox 为空——这种 **proactive turn** 仍会跑 triage：把房间最近的消息尾巴交给小模型，让它判断「是否仍有人欠话」，沉默是合法答案；agent-only loop cap 照常兜底。资格判定全程是算术（年龄 / 作者 / 读位），不含内容分类。

## 崩溃自愈：claim 的 TTL 抢占

claim（`one-of-us` 任务锁）的正常释放有两条路：赢家回复落地，或赢家未履约时 turn 收束前代码 `release_claim`。但 turn 中途**崩溃**的进程没有收束——赢来的锁会把 `task_key` 钉死，任务永远没人能再领。`try_claim` 因此带一条泄压阀：claim 超过 `CLAIM_TTL_S`（默认 300s）未动，任何 agent 一条 `ON CONFLICT … DO UPDATE WHERE created_at < now() - TTL` 原子抢走，不存在两个抢夺者各赢各的的窗口。被抢后原赢家在飞的回复仍会落地（正确性由 verbatim-dup 与 freshness 把守），它只是不再持锁；同一 agent 重复 claim 同一把钥匙是幂等刷新而非输。

开 K8s Job 宿主见 [k8s/README.md](k8s/README.md)。
