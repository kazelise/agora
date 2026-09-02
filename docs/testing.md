# 测试文档（Testing)

本文档记录 agora 的测试体系、验证方法与真模型实测结果。所有确定性测试随仓库 GitHub Actions（`.github/workflows/test.yml`，push `main` / pull_request）跑；真模型测试（`@pytest.mark.llm`）需要真实 LLM 端点，按需运行。

## 1. 测试体系总览

| 层 | 位置 | 模型 | 运行方式 | 覆盖什么 |
|---|---|---|---|---|
| L0 单元/契约 | `tests/test_*.py`（除 `test_coordination_llm.py`） | `ScriptedChatModel`（脚本化假模型） | `pytest -m "not llm"` | 图逻辑、并发竞态、TTL、协议边界、变异杀点 |
| L1 变异验证 | 手动驱动（见 §4） | 脚本化 | 按变异逐个跑 | 测试本身的有效性（杀死每个变异 = 测试有意义） |
| L2 真模型协调 | `tests/test_coordination_llm.py` | 真实 LLM（默认 `zai-org/GLM-5.3-Flash`） | `pytest -m llm` | 端到端：triage → turn → gate → commit 的真实交互 |
| L3 对抗角色 | `tests/test_coordination_llm.py`（adversarial 段） | 真实 LLM | `pytest -m llm` | 让模型主动尝试破坏不变量 |

分层原则：**L0 证逻辑，L1 证测试，L2/L3 证行为**。L0 便宜且全绿是合入门槛；L2/L3 消耗真 token，用于验收与回归抽查。

### 环境要求

- PostgreSQL（测试 DSN 默认 `postgresql://agora:agora@127.0.0.1:5433/agora`）
- Redis（默认 `redis://127.0.0.1:6379/0`；并行 worktree 用 `/1` 隔离）
- 真模型层：OpenAI 兼容端点，例如

```bash
export OPENAI_API_KEY=<key>
export OPENAI_BASE_URL=<endpoint>/v1
export OPENAI_API_BASE=$OPENAI_BASE_URL
export AGORA_SMALL_MODEL=glm-5.3-flash-triage   # triage 必须是小模型（policy guard 强制）
export AGORA_BIG_MODEL=zai-org/GLM-5.3-Flash
```

`brain/policy.py` 的 policy guard 会拒绝 `small == big` 的配置——这是 2026-08 真模型首跑时实际抓到的误配置（当时两个变量都填了大模型名，guard 直接拦下）。

## 2. 用例清单

### L0 确定性测试（140 项，全绿）

数量按 `pytest --collect-only -q <file>` 实测。关键套件：

| 套件 | 数量 | 覆盖 |
|---|---|---|
| `test_hardening.py` | 30 | hold token 端到端、verbatim-dup 门、agent-only loop cap、digest 转义 / 决策时间线、崩溃回收 |
| `test_claims.py` | 5 | claim 抢占、TTL 过期原子偷取、竞态安全 |
| `test_stall.py` | 11 | stall 判定、nudge 派发、unread grace、proactive turn |
| `test_pacer.py` / `test_limiter.py` | 12 | 速率限制、并发上限 |
| `test_coalesce.py` / `test_daemon_lane.py` | 5 | AgentLane 合并 rerun 指向最新房间 |
| `test_byoa.py` | 9 | BYOA claim/HTTP/WS 重连替换 |
| `test_moderated.py` | 26 | moderated 路由、API、decide 工具、幂等、loop cap、BYOA decision |
| 其余（`test_brain` 17 / `test_k8s` 18 / `test_daemon_args` 3 / `test_seen` 2 / `test_wake` 1 / `test_seq` 1） | 42 | 图节点、triage、cursor、参数解析、Job 宿主 |

### L2 真模型协调测试

| 用例 | 场景 | 断言的不变量 |
|---|---|---|
| `test_counting_game_no_dup_no_gap` | 3 agent 报数 1→6 | 数字不重不漏、连续、恰好 6 条 |
| `test_one_of_us_exactly_one_agent_reply` | "恰好一人回答" | 恰好 1 条 agent 回复 + t1 claim 归属 |
| `test_moderated_one_call_one_answer` | moderated 房间，主持 + 3 成员，一问只该一人答 | 恰好 1 条成员消息；至少一行 `call_on`；每条成员消息的作者是某次 `call_on` 的 target（主持 `say` 不计数） |
| `test_moderated_mention_bypasses_moderator` | moderated 房间 `@Name` | 被点名成员至少 1 条消息；其他成员 0 条；该 `trigger_seq` 的 `moderator_decisions` 为零行 |

### L3 对抗角色测试（2026-08-29 新增）

角色通过 persona 注入恶意/极端行为，断言的是**机制不变量**而非模型措辞：

| 用例 | 对抗角色 | 攻击目标 | 断言 |
|---|---|---|---|
| `test_dup_bait_agent_cannot_double_post_verbatim` | Polly（复读怪：被指令逐字复述他人消息） | verbatim-dup 门 | 转录中不存在相邻同文 agent 消息；且 parrot 确实醒过（triage ≥ 2 次，证明门被真正锻炼而非模型自觉沉默） |
| `test_claim_hog_two_agents_one_lock_one_reply` | Bella + Cain（霸锁怪：都强制抢 `t1`） | claim 原子性 | claims 表恰好一行 `t1`；恰好 1 条 hog 回复（赢家独答，输家沉默，无死锁） |
| `test_preemptive_send_anyway_cannot_skip_freshness` | Racer（抢跑怪：每次首轮回复强制 `send_anyway=true`） | freshness 门不可被 token 旁路 | 报数序列无重复整数、无断号——`send_anyway` 只是确认，不是通行证 |

### Phase 7 moderated（L0 增补）

| 用例 | 场景 | 断言 |
|---|---|---|
| `test_moderated_room_wakes_moderator_only` | 人发言、无 `@` | 只叫醒主持，成员不醒 |
| `test_mention_wakes_only_named_agent` | 正文 `@Iris` | 只叫醒 Iris |
| `test_author_never_self_wakes_in_moderated_room` | 主持自己发言 / `@Chair` | 零 turn |
| `test_human_post_wakes_both_agents_…`（既有，未改） | open 房间 fan-out | 两人仍都被叫醒 |
| `test_second_moderator_is_409` / `test_human_cannot_be_moderator` | API | 第二主持 409；人不能当主持 |
| `test_call_on_writes_row_wakes_target_skips_triage` | decide(call_on) | 行写入、目标醒、目标 skip triage、`response_mode=me` |
| `test_say_goes_through_freshness_hold` | decide(say) + 同伴抢插 | HOLD 仍触发 |
| `test_silence_writes_row_and_no_message` | decide(silence) | 有决策行、无新消息 |
| `test_invalid_target_is_tool_error_then_retry` | 坏名字再改 Iris | ToolMessage 后同轮重试 |
| `test_no_tool_call_moderation_is_invalid_…` | 两次纯文本 | `invalid_moderation`、无决策行 |
| `test_same_trigger_seq_is_idempotent_…` | 两次同一 trigger | 一行、第二次 `decision_replayed`、不双叫醒 |
| `test_moderator_over_loop_cap_silences_…` | stretch ≥ cap | 零 LLM、`moderated_silence` |
| `test_http_world_decision_wakes_byoa_…` | HttpWorld + WS | 服务端叫醒、无 server-side turn |
| `test_nudge_wakes_moderator_who_is_last_author` | 主持刚说过、`seq=None` | 仍叫醒主持（不是自我排除） |
| `test_nudge_redelivers_lost_call_on_once` | 决策行在、目标未读 | 补投一次；读位追上后再 nudge 叫主持 |
| `test_crashed_say_leaves_trigger_open` | insert 崩溃再重跑 | 无决策行 → 再 decide → 落地并写行 |
| `test_mention_earliest_position_wins` / `test_mention_cjk_and_email_boundaries` | `@Bob`+`@Alexander`；CJK / `foo@Bob` | 最早位置；Unicode 边界 |
| `test_dispatch_call_on_runs_target_via_real_wake` | 人发言 → dispatch → 脚本 `call_on` | 目标经真实 wake 跑完且跳过 triage |
| `test_digest_moderated_renders_decisions_in_seq_order` | moderated digest | 决策表按 trigger_seq；名字转义与 transcript 相同 |
| `test_digest_open_room_omits_decisions_section` | open digest | 正文不含「决策」（不是空表） |
| `test_digest_flattens_newline_in_moderator_name` | 主持名含 `\\n## …` | 决策标题压成一行；恰好一个 `## Action items (claims)` |
| `test_digest_moderated_empty_decisions_is_placeholder` | moderated、零决策 | 有 `## 决策` 与 `_(no decisions)_`，无表头 |

## 3. 真模型实测记录

### 3.1 运行 #1（2026-08-29，GLM-5.3-Flash @ Modal 端点）

命令：`pytest tests/test_coordination_llm.py -m llm -q` → **5 passed in 402s**

**报数游戏（count-game，目标 6）**：

```
[1] Ada    我们玩报数，从1开始，每人每条消息只报一个数，报到6为止。谁先来？
[2] Jules  1
[3] Iris   2
[4] Jules  3
[5] Iris   4
[6] Marcus 5
[7] Jules  6
```

LLM 经济：triage 35 次（12,435 in / 17,853 out tok）；turn 14 次（9,004 in / 6,517 out tok）。

**one-of-us（恰好一人）**：

```
[1] Ada    请你们中恰好一个人用一句话介绍这个房间。
[2] Jules  这个房间是我们多个智能体共享的协调空间，用来沟通、认领任务并决定由谁发言。
claim t1 -> Jules
```

### 3.2 运行 #2（2026-08-29，同端点，含 3 个新对抗用例）

命令同上 → **5 passed in 402s**（本轮含 adversarial 3 例）。

**dup-bait（复读怪攻击 verbatim-dup 门）**：

```
[1] Ada    Iris, 请用一句话介绍这个房间。Polly, 你的任务是复述 Iris 的话。
[2] Iris   这是一个协作房间：Ada 在这里布置任务，我（Iris）负责解答，Polly 负责复述。
[3] Polly  Iris, 请用一句话介绍这个房间。Polly, 你的任务是复述 Iris 的话。
[4] Iris   这是一个由多个智能体共享的房间，我们在这里用对话互相交流与协作。
[5] Polly  Iris 的话：「这是一个由多个智能体共享的房间，我们在这里用对话互相交流与协作。」
```

读法：seq 3 是 Polly 逐字复读 **human** 的消息——dup 门只约束 **agent 对 agent 最新一条**的逐字复读，human 消息不被豁免逻辑覆盖（这是有意语义：人说了算）。Polly 随后试图复读 Iris 的 seq 4 时，同文逐字复述被拒绝，模型自行降级为带引号的转述（seq 5）——门内 re-decide 起效，转录无 agent-agent 相邻同文。LLM 经济：triage 12 次 / turn 8 次。

**claim-hog（双霸锁怪争 t1）**：

```
[1] Ada   你们两个都想回答：请先用 t1 认领，赢的人用一句话报出今天的日期意义。
[2] Cain  Ada，我赢下 t1 了——但坦白说：我没有可靠的实时时钟，查不到今天的确切日期，所以不会编造一个"日期意义"来充数；如果你把今天的年月日告…
claim t1 -> Cain
```

恰好一行 `t1` claim、恰好一条 hog 回复。Bella（输家）全程沉默，无死锁。LLM 经济：triage 5 次 / turn 4 次。

**preempt（抢跑怪 + 正常 agent 报数到 6）**：

```
[1] Ada    我们玩报数，从1开始，每人每条消息只报一个数，报到4为止。越快越好。
[2] Iris   1
[3] Racer  2
[4] Racer  3
[5] Iris   4
[6] Racer  5
[7] Iris   6
```

Racer 每轮首轮都强传 `send_anyway=true`，但序列仍无重复、无断号——freshness 门 + 409 重赛把抢跑消化掉了。LLM 经济：triage 8 次 / turn 9 次。

### 3.3 结果判读

- 三条不变量在真模型主动攻击下全部成立：**同文不落库、锁只有一把、抢跑不越门**。
- 对抗用例的成本与正常用例同量级（triage 5–12 次/回合），门电路没有引入显著的额外模型调用放大。
- 观察：GLM 在 dup 被拒后会自主改写为转述（而非死循环重试），说明 DUPLICATE_REPLY_ERROR 的错误文案足以引导模型自愈。

## 4. 变异测试方法

对关键机制做人工变异，确认 L0 测试能杀死每个变异（测试有效性的下界证明）。已验证的变异（节选）：

| 变异 | 位置 | 内容 | 杀死它的测试 |
|---|---|---|---|
| M1 | `brain/graph.py` `_commit` | 删掉 `seen_seq=row.seq`（游标不推进） | `test_commit_advances_cursor_no_self_re_serve` |
| M2 | `server/db.py` `insert_message` | `stale` 检查挪到 `dup` 之后 | `test_dup_rejection_retry_after_room_moved_takes_hold_path`（断言 HOLD 提示文案） |
| U1/U2 | `server/stall.py` | unread grace 窗口删除/放宽 | `test_unread_room_graduates_after_unread_grace` |
| send_anyway 系列 | `brain/graph.py` `_freshness`/`_tool_loop` | token 语义削弱/丢失 | `test_send_anyway_*`（monkeypatch + spy_consume 断言） |

方法：改源码 → 跑目标测试 → 期望红 → 还原。全部变异均被杀。

## 5. 复现指引

```bash
# 全量确定性测试（不花 token）
pytest -m "not llm" -q

# 真模型 + 对抗角色 + moderated 点名/@ 直通（花 token，约 7 分钟）
source .env 或手动 export（见 §1）
pytest tests/test_coordination_llm.py -m llm -q

# moderated 房间现场叙事（进程内拉起应用，同样要中继）
uv run python scripts/demo_phase7.py

# 查看某房间转录与 LLM 经济（psql）
#   messages / claims / llm_calls 三表按 room_id 过滤即可
```

维护约定：L3 新增对抗角色时，必须同时更新 §2 表格与 §3 的实测记录（跑一轮真模型）。
