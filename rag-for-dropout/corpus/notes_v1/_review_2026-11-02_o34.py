"""One-off: apply the author's O34 decision (2026-11-02): accept the 11 notes that the O30 review did not mention (set status reviewed); the SOP stays draft.
    .venv/bin/python rag-for-dropout/corpus/notes_v1/_review_2026-11-02_o34.py
Only the front matter changes; the text is not touched (a test checks the golden spans)."""
from pathlib import Path

import yaml

NOTES = Path(__file__).resolve().parent
ACCEPT = ["note_08_partial_observability", "note_10_window_semantics", "note_12_missing_data_representation", "note_13_dropout_aware_training",
          "note_17_noise_versus_dropout", "note_18_soft_vote", "note_23_similar_simulated_cases", "note_24_gate_outcomes", "note_27_retrieval_scores",
          "note_29_gate_belongs_to_one_checkpoint", "note_30_aggregate_versus_individual"]
for stem in ACCEPT:
    p = NOTES / f"{stem}.md"
    raw = p.read_text(encoding="utf-8")
    end = raw.index("\n---\n", 4)
    front, body = yaml.safe_load(raw[4:end]), raw[end + 5:]
    assert front["status"] == "draft", stem
    front.update(status="reviewed", version=1, reviewed="2026-11-02", review="accepted by the author in the O34 decision (2026-11-02)",
                 verification="author review O34 2026-11-02; source_ref names project files")
    p.write_text("---\n" + yaml.safe_dump(front, sort_keys=False, allow_unicode=True, width=1000) + "---\n" + body, encoding="utf-8")
print(len(ACCEPT), "notes set to reviewed")
