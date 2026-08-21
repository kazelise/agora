Phase 2 的 LangGraph 大脑：`triage` → `tool_loop` → `freshness` HOLD → `commit`。

设计理由写在 [docs/design.md](../docs/design.md) 的 Phase 2 一节。入口是 `brain.graph.Brain` / `make_turn_fn`。
