LangGraph 大脑：`triage` → `tool_loop` → `freshness` HOLD → `commit`（事务内再比一次 `seen_seq`）。`claim` 的 `task_key` 必须锚定触发消息的 seq。副作用只走 `World`（`DirectWorld` 云端 / `HttpWorld` daemon），图本身不碰 Postgres 或 Redis。

设计理由写在 [docs/design.md](../docs/design.md) 的 Phase 2 / Phase 4b 两节。入口是 `brain.graph.Brain` / `make_turn_fn`。
