#!/bin/sh
# One container, two processes: the mock HR system (port 9000, internal) and the public API (port $PORT).
uvicorn mock_services.mock_server:app --host 127.0.0.1 --port 9000 &
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
