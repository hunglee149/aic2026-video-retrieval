"""Gunicorn config — Azure Container App (4 vCPU / 8 GiB).

Chạy bằng:
    gunicorn aic.ui.app:app -c gunicorn.conf.py

1 worker duy nhất: bắt buộc để WebSocket sync hoạt động đúng.
In-memory shared state (_shared_state, _ws_clients) không thể chia sẻ
giữa các process riêng biệt. Với 5 người dùng cuộc thi, 1 uvicorn worker
+ async event loop là đủ — FAISS/CLIP chạy trong thread pool của FastAPI.
"""

import os

# ---------------------------------------------------------------------------
# Worker processes
# ---------------------------------------------------------------------------

# 1 worker duy nhất: bắt buộc để WebSocket sync hoạt động đúng.
# In-memory shared state (_shared_state, _ws_clients) không thể chia sẻ
# giữa các process riêng biệt. Với 5 người dùng cuộc thi, 1 uvicorn worker
# + async event loop là đủ — FAISS/CLIP chạy trong thread pool của FastAPI.
# Nếu cần scale sau này: dùng Redis làm shared state thay vì in-memory.
workers = 1

# UvicornWorker: async event loop xử lý WebSocket + HTTP đồng thời.
worker_class = "uvicorn.workers.UvicornWorker"

# Thread pool cho các blocking call trong sync def endpoints (search, translate).
worker_connections = 100

# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------

# CLIP inference trên CPU có thể mất 5-10 giây.
# 120s đủ cho cả cold-start retrieval lần đầu.
timeout = 120

# Giữ kết nối HTTP/1.1 alive để giảm overhead TCP handshake.
keepalive = 5

# Graceful shutdown: cho phép request hiện tại hoàn thành trước khi tắt.
graceful_timeout = 30

# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------

port = os.environ.get("PORT", "8000")
bind = f"0.0.0.0:{port}"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

accesslog = "-"   # stdout
errorlog  = "-"   # stderr
loglevel  = os.environ.get("LOG_LEVEL", "info")

# Ghi rõ worker PID trong log để phân biệt luồng khi debug.
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" worker=%(p)s'

# ---------------------------------------------------------------------------
# Process title (hiển thị đẹp trong `ps aux` trên Azure)
# ---------------------------------------------------------------------------

proc_name = "aic-video-retrieval"

# ---------------------------------------------------------------------------
# Lifecycle hooks
# ---------------------------------------------------------------------------

def on_starting(server):
    server.log.info("Gunicorn master starting — workers=%d", workers)


def post_fork(server, worker):
    """Sau khi fork: mỗi worker load model độc lập để tránh shared state."""
    server.log.info("Worker %s spawned (pid: %s)", worker.age, worker.pid)
