# Phase 4c: cloud turns as Kubernetes Jobs

`computer_id` 为空的 Agent 默认仍在 API 进程里跑（`DirectWorld`）。打开 `AGORA_K8S_ENABLED=1` 之后，调度器的车道函数换成「创建一个 Job 并等它结束」。图还是那张；Job 用 cluster token 打 `/runtime/*`（`HttpWorld`），回复走服务端 fan-out。

车道仍在服务端：突发叫醒还是「在飞的一轮 + 再一轮」。`backoffLimit: 0`——k8s 重试不是 turn 重试，脏标记才是。

Job 创建失败或 Job 失败是 miss 一轮（fail-open），不回退到进程内，免得把配置错误藏起来。

## 本地 kind

Postgres / Redis / uvicorn 仍在宿主机（`docker compose` + `uv run uvicorn`）。kind 只跑 turn Job。

```bash
kind create cluster --config k8s/kind.yaml
docker build -t agora:local .
kind load docker-image agora:local
kubectl apply -f k8s/namespace.yaml -f k8s/rbac.yaml
# 按 k8s/secret.example.yaml 填一份本地 secret 再 apply
```

Job 要打到宿主机上的 API。Docker Desktop 用 `http://host.docker.internal:8000`；Linux 上换成 kind 节点能路由到的宿主机地址（常见是 docker0 / 默认网关）。

```bash
export AGORA_K8S_ENABLED=1
export AGORA_CLUSTER_TOKEN=the-same-token-as-the-secret
export AGORA_K8S_API_URL="$(kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}')"
export AGORA_K8S_TOKEN="$(kubectl --namespace agora create token agora-scheduler)"
export AGORA_K8S_INSECURE=1
export AGORA_K8S_SERVER_URL=http://host.docker.internal:8000
# 生产形态：export AGORA_K8S_SECRET_NAME=agora-cloud
uv run uvicorn server.main:app --reload --port 8000
```

往一间有云端 Agent 的房间 POST 一条人的消息，然后：

```bash
kubectl --namespace agora get jobs
kubectl --namespace agora logs -l agora.role=turn
```

## 镜像

仓库根目录的 `Dockerfile` 默认入口是 `python -m brain.job`。服务端另写 command 即可共用同一张镜像。
