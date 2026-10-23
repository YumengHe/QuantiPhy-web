#!/usr/bin/env bash
# Serve this directory on port $PORT (default 9002) in the background (tools/serve.py: http.server + Range, so videos can seek). Usage: ./serve.sh [start|stop|status]
cd "$(dirname "$0")"
PORT=${PORT:-9002}
PIDFILE=.serve.pid
case "${1:-start}" in
  start)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then echo "already running (pid $(cat $PIDFILE))"; exit 0; fi
    nohup python3 tools/serve.py "$PORT" 0.0.0.0 > .serve.log 2>&1 &
    echo $! > "$PIDFILE"; echo "started on http://$(hostname -I | awk '{print $1}'):$PORT (pid $!)";;
  stop)
    [ -f "$PIDFILE" ] && kill "$(cat $PIDFILE)" 2>/dev/null && rm -f "$PIDFILE" && echo stopped || echo "not running";;
  status)
    [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null && echo "running (pid $(cat $PIDFILE))" || echo "not running";;
esac
