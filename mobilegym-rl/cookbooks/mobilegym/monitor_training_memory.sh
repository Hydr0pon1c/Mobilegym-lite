#!/usr/bin/env bash
# Copyright 2026 OPPO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

OUTPUT_DIR="${1:?usage: monitor_training_memory.sh OUTPUT_DIR [INTERVAL_SECONDS]}"
INTERVAL_SECONDS="${2:-5}"
CGROUP_MEMORY_DIR="/sys/fs/cgroup/memory"
mkdir -p "$OUTPUT_DIR"

SUMMARY_FILE="$OUTPUT_DIR/host_memory.csv"
PROCESS_FILE="$OUTPUT_DIR/process_snapshots.log"
PPU_FILE="$OUTPUT_DIR/ppu_snapshots.log"

if [[ ! -s "$SUMMARY_FILE" ]]; then
  echo "timestamp,cgroup_usage_bytes,cgroup_max_usage_bytes,cgroup_limit_bytes,cgroup_failcnt,cgroup_oom_kill,cgroup_rss_bytes,cgroup_cache_bytes,mem_available_bytes,shm_used_bytes,gateway_db_bytes" > "$SUMMARY_FILE"
fi

read_value() {
  local path="$1"
  if [[ -r "$path" ]]; then
    tr -d '\n' < "$path"
  else
    echo 0
  fi
}

read_stat_value() {
  local key="$1"
  local path="$2"
  awk -v wanted="$key" '$1 == wanted { print $2; found=1; exit } END { if (!found) print 0 }' "$path" 2>/dev/null
}

while true; do
  timestamp="$(date --iso-8601=seconds)"
  usage="$(read_value "$CGROUP_MEMORY_DIR/memory.usage_in_bytes")"
  max_usage="$(read_value "$CGROUP_MEMORY_DIR/memory.max_usage_in_bytes")"
  limit="$(read_value "$CGROUP_MEMORY_DIR/memory.limit_in_bytes")"
  failcnt="$(read_value "$CGROUP_MEMORY_DIR/memory.failcnt")"
  oom_kill="$(read_stat_value oom_kill "$CGROUP_MEMORY_DIR/memory.oom_control")"
  rss="$(read_stat_value total_rss "$CGROUP_MEMORY_DIR/memory.stat")"
  cache="$(read_stat_value total_cache "$CGROUP_MEMORY_DIR/memory.stat")"
  mem_available="$(awk '$1 == "MemAvailable:" { print $2 * 1024 }' /proc/meminfo)"
  shm_used="$(df --output=used -B1 /dev/shm | tail -n 1 | tr -d ' ')"
  gateway_db_bytes="$(find ${RLLM_HOME:-$HOME/.rllm} -maxdepth 1 -name 'gateway_traces.db*' -type f -printf '%s\n' 2>/dev/null | awk '{ total += $1 } END { print total + 0 }')"

  echo "$timestamp,$usage,$max_usage,$limit,$failcnt,$oom_kill,$rss,$cache,$mem_available,$shm_used,$gateway_db_bytes" >> "$SUMMARY_FILE"
  {
    echo "===== $timestamp cgroup_usage=$usage cgroup_rss=$rss cgroup_cache=$cache ====="
    # Never persist command lines here: platform mount processes may expose
    # access credentials in their argv.  The process name is enough to
    # attribute memory growth to Ray, vLLM, Chromium, or the gateway.
    ps -eo pid=,ppid=,rss=,vsz=,%cpu=,stat=,comm= --sort=-rss | head -n 35 || true
  } >> "$PROCESS_FILE"
  {
    echo "===== $timestamp ====="
    nvidia-smi || true
  } >> "$PPU_FILE" 2>&1
  sleep "$INTERVAL_SECONDS"
done
