BYOA daemon：同一张 LangGraph，用户自己的模型 key，服务端不持有凭据。

```bash
uv run python -m daemon \
  --server http://127.0.0.1:8000 \
  --computer-id "$AGORA_COMPUTER_ID" \
  --token "$AGORA_COMPUTER_TOKEN"
```

环境变量备选：`AGORA_SERVER_URL` / `AGORA_COMPUTER_ID` / `AGORA_COMPUTER_TOKEN`。
`OPENAI_API_KEY` 和 `OPENAI_BASE_URL` 只在这个进程里读，启动时会打 base_url 和模型名，不打 key。
