#!/bin/bash
# GPU 작업 순차 실행 데몬 (GPU 경합 시 처리량이 크게 떨어지므로 한 번에 하나만 실행)
# ../jobs/queue.txt 에 줄을 추가하면 위에서부터 차례로 실행:  "<name> <script.py> <인자...>"
#   완료 표시: results/runs/<name>.json | <name>_s0.json | teachers/<name>.pt
#   실패 표시: results/logs/<name>.fail  (재시도 안 함)
# 큐가 비면 10초마다 다시 확인. ../jobs/STOP 파일이 생기면 종료.
cd "$(dirname "$0")"
PY=../../.venv/bin/python
R=../results
while [ ! -f ../jobs/STOP ]; do
  job=""
  while read -r name script rest; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    if [ -f $R/runs/$name.json ] || [ -f $R/runs/${name}_s0.json ] || [ -f $R/logs/$name.fail ]; then continue; fi
    job="$name $script $rest"; break
  done < ../jobs/queue.txt
  if [ -z "$job" ]; then sleep 10; continue; fi
  read -r name script rest <<< "$job"
  echo "$(date +%T) START $name"
  $PY $script --name $name $rest > $R/logs/$name.log 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    touch $R/logs/$name.fail
    echo "$(date +%T) FAIL $name rc=$rc"
  else
    echo "$(date +%T) $(tail -1 $R/logs/$name.log | cut -c1-300)"
  fi
done
echo "daemon stopped"
