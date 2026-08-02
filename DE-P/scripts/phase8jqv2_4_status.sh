#!/usr/bin/env bash
set -euo pipefail
cd /home/zjh/YOPO/DE-P
conda run --no-capture-output -n yopo python - <<'PY'
import json, os, time
from collections import Counter
from pathlib import Path
import yaml

root=Path("data/phase8_authoritative_v1")
cfg=yaml.safe_load(Path("configs/phase8_authoritative_v1_generation.yaml").read_text())
journal=root/"generation_state/generation_journal.jsonl"
events=[]
if journal.is_file():
    for line in journal.read_text().splitlines():
        try: events.append(json.loads(line))
        except json.JSONDecodeError: pass
complete=[e for e in events if e.get("event")=="sequence_complete"]
maps=[e for e in events if e.get("event")=="map_complete"]
failed=[e for e in events if e.get("status")=="failed"]
frames=sum(e.get("frames",0) for e in complete)
tot=sum(cfg["formal_splits"][s]["frames"] for s in ("train","valid"))
start=min([e["timestamp_ns"] for e in events] or [time.time_ns()])
elapsed=max((time.time_ns()-start)/1e9,1e-6)
rate=frames/elapsed
current=next((e for e in reversed(events) if e.get("status")=="running"),None)
locks=list((root/"generation_state/locks").glob("*.lock")) if root.exists() else []
global_lock=root/"generation_state/phase8jqv2_4_generation.lock"
print(json.dumps({
 "overall":"COMPLETE" if (root/"generation_state/completion/FULL_GENERATION_COMPLETE").is_file() else ("RUNNING" if global_lock.is_file() else "INCOMPLETE"),
 "train_maps":f"{sum(e.get('split')=='train' for e in maps)}/{len(cfg['formal_splits']['train']['maps'])}",
 "valid_maps":f"{sum(e.get('split')=='valid' for e in maps)}/{len(cfg['formal_splits']['valid']['maps'])}",
 "static_sequences_complete":sum(e.get("suite")=="static" for e in complete),
 "dynamic_sequences_complete":sum(e.get("suite")=="dynamic" for e in complete),
 "frames":f"{frames}/{tot}", "certificates":f"{frames}/{tot}",
 "failed_work_units":len(failed), "current_work_unit":current,
 "elapsed_hours":elapsed/3600, "throughput_frames_s":rate,
 "estimated_remaining_hours":(tot-frames)/rate/3600 if rate else None,
 "disk_usage_bytes":sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0,
 "locks":[str(p) for p in locks]+([str(global_lock)] if global_lock.is_file() else []),
 "completion_marker":(root/"generation_state/completion/FULL_GENERATION_COMPLETE").is_file(),
},indent=2))
PY
conda run --no-capture-output -n yopo python - <<'PY' \
  >> logs/phase8jqv2_4/resource_usage.jsonl
import json, os, time
from pathlib import Path
root=Path("data/phase8_authoritative_v1")
print(json.dumps({
 "timestamp_ns":time.time_ns(),
 "disk_bytes":sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
 if root.exists() else 0,
 "load_average":os.getloadavg(),
}))
PY
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
