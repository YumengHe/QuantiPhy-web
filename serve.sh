#!/usr/bin/env bash
# Serve this directory on port 9688 in the background. Usage: ./serve.sh [start|stop|status]
cd "$(dirname "$0")"
PORT=9688
PIDFILE=.serve.pid
case "${1:-start}" in
  start)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then echo "already running (pid $(cat $PIDFILE))"; exit 0; fi
    nohup python3 -m http.server "$PORT" --bind 0.0.0.0 > .serve.log 2>&1 &
    echo $! > "$PIDFILE"; echo "started on http://$(hostname -I | awk '{print $1}'):$PORT (pid $!)";;
  stop)
    [ -f "$PIDFILE" ] && kill "$(cat $PIDFILE)" 2>/dev/null && rm -f "$PIDFILE" && echo stopped || echo "not running";;
  status)
    [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null && echo "running (pid $(cat $PIDFILE))" || echo "not running";;
esac
