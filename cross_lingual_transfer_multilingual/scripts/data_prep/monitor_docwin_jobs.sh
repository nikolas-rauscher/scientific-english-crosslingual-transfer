#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Monitor running Slurm jobs and map JobID -> language/split/progress for docwin runs.

Usage:
  bash cross_lingual_transfer_multilingual/scripts/data_prep/monitor_docwin_jobs.sh [options]

Options:
  --user USER             Slurm user (default: $USER)
  --project-root PATH     Project root (default: /netscratch/anonymous_user/projects/BA-hydra)
  --job-name NAME         Filter by exact Slurm job name (default: clt_multi_docwin)
                          Use "all" to show all jobs from the user.
  --interval SEC          Refresh interval in seconds (default: 30)
  --once                  Print one snapshot and exit.
  -h, --help              Show this help.

Examples:
  # watch only docwin jobs
  bash cross_lingual_transfer_multilingual/scripts/data_prep/monitor_docwin_jobs.sh

  # one-shot snapshot
  bash cross_lingual_transfer_multilingual/scripts/data_prep/monitor_docwin_jobs.sh --once

  # include all user jobs (including bloom eval)
  bash cross_lingual_transfer_multilingual/scripts/data_prep/monitor_docwin_jobs.sh --job-name all
EOF
}

USER_NAME="${USER:-}"
PROJECT_ROOT="/netscratch/anonymous_user/projects/BA-hydra"
JOB_NAME_FILTER="clt_multi_docwin"
INTERVAL_SEC=30
ONCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user)
      USER_NAME="${2:?missing value for --user}"
      shift 2
      ;;
    --project-root)
      PROJECT_ROOT="${2:?missing value for --project-root}"
      shift 2
      ;;
    --job-name)
      JOB_NAME_FILTER="${2:?missing value for --job-name}"
      shift 2
      ;;
    --interval)
      INTERVAL_SEC="${2:?missing value for --interval}"
      shift 2
      ;;
    --once)
      ONCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v squeue >/dev/null 2>&1; then
  echo "ERROR: squeue not found. Run this script on a Slurm login node." >&2
  exit 1
fi

if [[ -z "$USER_NAME" ]]; then
  echo "ERROR: no user set. Pass --user USER." >&2
  exit 1
fi

if ! [[ "$INTERVAL_SEC" =~ ^[0-9]+$ ]] || [[ "$INTERVAL_SEC" -lt 1 ]]; then
  echo "ERROR: --interval must be an integer >= 1" >&2
  exit 1
fi

LOG_DIR_MULTI="$PROJECT_ROOT/cross_lingual_transfer_multilingual/logs"
LOG_DIR_LEGACY="$PROJECT_ROOT/cross_lingual_transfer/logs"

find_log_for_job() {
  local jid="$1"
  local job_name="$2"
  local candidate=""

  candidate="$LOG_DIR_MULTI/slurm_${job_name}_${jid}.out"
  if [[ -f "$candidate" ]]; then
    echo "$candidate"
    return 0
  fi

  candidate="$(find "$LOG_DIR_MULTI" "$LOG_DIR_LEGACY" -maxdepth 1 -type f -name "*${jid}*.out" 2>/dev/null | head -n1 || true)"
  if [[ -n "$candidate" ]]; then
    echo "$candidate"
    return 0
  fi

  echo ""
  return 0
}

extract_lang_from_log() {
  local log="$1"
  local lang=""

  lang="$(grep -m1 -E "^Language:[[:space:]]|Running WECHSEL for|Selected languages:" "$log" 2>/dev/null || true)"
  lang="$(echo "$lang" | sed -E \
    -e 's/^Language:[[:space:]]*//' \
    -e 's/.*Running WECHSEL for[[:space:]]*//' \
    -e 's/[[:space:]]*->.*//' \
    -e 's/.*Selected languages:[[:space:]]*//')"
  [[ -z "$lang" ]] && lang="n/a"
  echo "$lang"
}

parse_progress_line() {
  local line="$1"
  local split completed total elapsed out_mb

  split="$(echo "$line" | sed -n 's/.*\[progress\]\[\([^]]*\)\].*/\1/p')"
  completed="$(echo "$line" | sed -n 's/.*completed_tasks=\([0-9][0-9]*\)\/\([0-9][0-9]*\).*/\1/p')"
  total="$(echo "$line" | sed -n 's/.*completed_tasks=\([0-9][0-9]*\)\/\([0-9][0-9]*\).*/\2/p')"
  elapsed="$(echo "$line" | sed -n 's/.*elapsed=\([0-9][0-9]*\)s.*/\1/p')"
  out_mb="$(echo "$line" | sed -n 's/.*output_size_mb=\([0-9][0-9]*\).*/\1/p')"

  echo "${split:-n/a}|${completed:-0}|${total:-0}|${elapsed:-0}|${out_mb:-0}"
}

format_duration() {
  local secs="$1"
  if ! [[ "$secs" =~ ^[0-9]+$ ]]; then
    echo "n/a"
    return
  fi
  local h=$((secs / 3600))
  local m=$(((secs % 3600) / 60))
  local s=$((secs % 60))
  printf "%02d:%02d:%02d" "$h" "$m" "$s"
}

estimate_eta_from_log() {
  local log="$1"
  local split="$2"
  local progress_lines first_line last_line
  local first_vals last_vals c0 c1 t0 t1 total remaining dt dc eta_sec

  progress_lines="$(grep -E "\\[progress\\]\\[$split\\]" "$log" 2>/dev/null || true)"
  [[ -z "$progress_lines" ]] && { echo "n/a"; return; }

  first_line="$(echo "$progress_lines" | head -n1)"
  last_line="$(echo "$progress_lines" | tail -n1)"
  first_vals="$(parse_progress_line "$first_line")"
  last_vals="$(parse_progress_line "$last_line")"

  c0="$(echo "$first_vals" | cut -d'|' -f2)"
  total="$(echo "$last_vals" | cut -d'|' -f3)"
  t0="$(echo "$first_vals" | cut -d'|' -f4)"
  c1="$(echo "$last_vals" | cut -d'|' -f2)"
  t1="$(echo "$last_vals" | cut -d'|' -f4)"

  if [[ "$total" -le 0 ]]; then
    echo "n/a"
    return
  fi

  if [[ "$c1" -ge "$total" ]]; then
    echo "00:00:00"
    return
  fi

  dc=$((c1 - c0))
  dt=$((t1 - t0))
  remaining=$((total - c1))

  if [[ "$dc" -le 0 ]] || [[ "$dt" -le 0 ]]; then
    echo "n/a"
    return
  fi

  # Integer math for eta seconds: remaining / (dc/dt) == remaining*dt/dc
  eta_sec=$(((remaining * dt) / dc))
  format_duration "$eta_sec"
}

print_snapshot() {
  local rows row_count=0
  local now
  now="$(date '+%Y-%m-%d %H:%M:%S')"

  mapfile -t rows < <(squeue -u "$USER_NAME" -h -o "%i|%j|%T|%P|%M|%l|%R")

  printf "Timestamp: %s\n" "$now"
  printf "Filter: user=%s job_name=%s\n" "$USER_NAME" "$JOB_NAME_FILTER"
  printf "%-8s  %-20s  %-10s  %-9s  %-10s  %-10s  %-7s  %-11s  %-8s  %-8s  %s\n" \
    "JOBID" "JOBNAME" "LANG" "STATE" "PART" "TIME" "SPLIT" "TASKS" "PCT" "ETA" "LOG"
  printf "%-8s  %-20s  %-10s  %-9s  %-10s  %-10s  %-7s  %-11s  %-8s  %-8s  %s\n" \
    "-----" "-------" "----" "-----" "----" "----" "-----" "-----" "---" "---" "---"

  for row in "${rows[@]}"; do
    local jid jname state part runtime timelimit node
    local log lang progress_line progress_vals split completed total elapsed out_mb tasks pct eta short_log

    jid="$(echo "$row" | cut -d'|' -f1)"
    jname="$(echo "$row" | cut -d'|' -f2)"
    state="$(echo "$row" | cut -d'|' -f3)"
    part="$(echo "$row" | cut -d'|' -f4)"
    runtime="$(echo "$row" | cut -d'|' -f5)"
    timelimit="$(echo "$row" | cut -d'|' -f6)"
    node="$(echo "$row" | cut -d'|' -f7)"

    if [[ "$JOB_NAME_FILTER" != "all" && "$jname" != "$JOB_NAME_FILTER" ]]; then
      continue
    fi

    row_count=$((row_count + 1))
    log="$(find_log_for_job "$jid" "$jname")"
    lang="n/a"
    split="n/a"
    completed="0"
    total="0"
    elapsed="0"
    out_mb="0"
    tasks="n/a"
    pct="n/a"
    eta="n/a"
    short_log="n/a"

    if [[ -n "$log" ]]; then
      lang="$(extract_lang_from_log "$log")"
      progress_line="$(grep -E "\\[progress\\]" "$log" 2>/dev/null | tail -n1 || true)"
      if [[ -n "$progress_line" ]]; then
        progress_vals="$(parse_progress_line "$progress_line")"
        split="$(echo "$progress_vals" | cut -d'|' -f1)"
        completed="$(echo "$progress_vals" | cut -d'|' -f2)"
        total="$(echo "$progress_vals" | cut -d'|' -f3)"
        elapsed="$(echo "$progress_vals" | cut -d'|' -f4)"
        out_mb="$(echo "$progress_vals" | cut -d'|' -f5)"
        tasks="${completed}/${total}"
        if [[ "$total" -gt 0 ]]; then
          pct="$(awk -v c="$completed" -v t="$total" 'BEGIN{printf "%.1f%%", (100*c)/t}')"
        fi
        eta="$(estimate_eta_from_log "$log" "$split")"
      fi
      short_log="${log#$PROJECT_ROOT/}"
    fi

    printf "%-8s  %-20s  %-10s  %-9s  %-10s  %-10s  %-7s  %-11s  %-8s  %-8s  %s\n" \
      "$jid" "$jname" "$lang" "$state" "$part" "$runtime/$timelimit" "$split" "$tasks" "$pct" "$eta" "$short_log"
  done

  if [[ "$row_count" -eq 0 ]]; then
    echo "No matching jobs found."
  fi
}

if [[ "$ONCE" -eq 1 ]]; then
  print_snapshot
  exit 0
fi

while true; do
  clear
  print_snapshot
  sleep "$INTERVAL_SEC"
done

