#!/usr/bin/env bash
# Kill v6_hardpairs eqbench3 + v8 eqbench3 (we only care about MMLU).
while true; do
    for pat in "eqbench3.*v6_hardpairs" "eqbench3.*v8_cfact"; do
        pids=$(pgrep -f "$pat" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "[$(date -u +%H:%M:%SZ)] killing $pat: $pids"
            kill -9 $pids 2>/dev/null || true
        fi
    done
    sleep 60
done
