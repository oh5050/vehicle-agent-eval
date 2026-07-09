import json

src = "data/final/dataset_final.backup.jsonl"
dst = "data/final/dataset_final.jsonl"

rows = [
    json.loads(l)
    for l in open(src, encoding="utf-8")
]

allowed_manual = set()

for prefix, count in [
    ("manual_T2_", 12),
    ("manual_T6_", 13),
    ("manual_T4_", 5),
    ("manual_T5_", 4),
]:
    for i in range(1, count + 1):
        allowed_manual.add(f"{prefix}{i:03d}")

seen = set()
keep = []

for r in rows:
    rid = r["id"]

    # 기존 검수 데이터
    if r.get("labeled_by") == "human":
        r["labeled_by"] = "llm_draft_reviewed"
        keep.append(r)

    # 필요한 manual만 첫 번째 1개 유지
    elif rid in allowed_manual:
        if rid not in seen:
            keep.append(r)
            seen.add(rid)

with open(dst, "w", encoding="utf-8") as f:
    for r in keep:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("final:", len(keep))