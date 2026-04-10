"""
CLIP Score Diagnostic Tool
---------------------------
Samples up to 100 random frames from your DB, calculates per-image CLIPScore,
detects token truncation, and exports a self-contained HTML report.

Run:
    pip install torch torchvision torchmetrics[multimodal] transformers pillow
    python clip_diagnostic.py
"""

import os
import random
import sqlite3
import base64
from io import BytesIO

import torch
from PIL import Image
import torchvision.transforms as T
from torchmetrics.multimodal.clip_score import CLIPScore
from transformers import CLIPTokenizer

# ─────────────────────────────────────────────
# CONFIG – adjust these to match your setup
# ─────────────────────────────────────────────
DB_PATH          = "output/surveillance.db"
FRAMES_DIR       = os.path.join(os.getcwd(), "output", "frames")
MODEL_NAME       = "openai/clip-vit-base-patch16"
SAMPLE_SIZE      = 100          # how many frames to sample
CLIP_TOKEN_LIMIT = 77           # CLIP's hard context window
HTML_OUT         = "clip_diagnostic_report.html"
THUMB_WIDTH      = 160          # px width for thumbnails in the report
# ─────────────────────────────────────────────


def encode_image_b64(pil_img: Image.Image, max_width: int = THUMB_WIDTH) -> str:
    """Resize and base64-encode a PIL image for embedding in HTML."""
    ratio = max_width / pil_img.width
    new_h = int(pil_img.height * ratio)
    thumb = pil_img.resize((max_width, new_h), Image.LANCZOS)
    buf = BytesIO()
    thumb.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def token_info(tokenizer: CLIPTokenizer, text: str) -> dict:
    """
    Returns token count and whether the text was truncated by CLIP's 77-token limit.
    Also returns the effective (post-truncation) text that CLIP actually sees.
    """
    # Encode WITHOUT truncation to see the real token count
    ids_full = tokenizer.encode(text, add_special_tokens=True)
    full_count = len(ids_full)
    truncated = full_count > CLIP_TOKEN_LIMIT

    # Decode what CLIP will actually use (first 77 tokens)
    if truncated:
        ids_trunc = ids_full[:CLIP_TOKEN_LIMIT]
        effective_text = tokenizer.decode(ids_trunc, skip_special_tokens=True)
    else:
        effective_text = text

    return {
        "full_token_count": full_count,
        "truncated": truncated,
        "tokens_used": min(full_count, CLIP_TOKEN_LIMIT),
        "tokens_dropped": max(0, full_count - CLIP_TOKEN_LIMIT),
        "effective_text": effective_text,
    }


def score_label(score: float) -> tuple[str, str]:
    """Human-readable quality label + CSS class for a score."""
    if score >= 30:
        return "Good", "good"
    elif score >= 22:
        return "Moderate", "moderate"
    else:
        return "Low", "low"


def main():
    print(f"[1/5] Connecting to DB: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            CASE
                WHEN clean_frame_path IS NOT NULL AND clean_frame_path != ''
                THEN clean_frame_path
                ELSE frame_path
            END AS img_path,
            analysis_text
        FROM analyses
        WHERE analysis_text IS NOT NULL
    """)
    all_records = cursor.fetchall()
    conn.close()

    if not all_records:
        print("No records found. Check your DB path and query.")
        return

    sample = random.sample(all_records, min(SAMPLE_SIZE, len(all_records)))
    print(f"[2/5] Sampled {len(sample)} records from {len(all_records)} total.")

    print(f"[3/5] Loading CLIP model + tokenizer: {MODEL_NAME}")
    clip_metric = CLIPScore(model_name_or_path=MODEL_NAME)
    tokenizer   = CLIPTokenizer.from_pretrained(MODEL_NAME)
    transform   = T.ToTensor()

    rows = []
    print(f"[4/5] Scoring frames…")

    for idx, (raw_path, text) in enumerate(sample):
        # Resolve path to local frames dir
        img_path = None
        if raw_path:
            fname    = os.path.basename(raw_path)
            img_path = os.path.join(FRAMES_DIR, fname)

        result = {
            "idx":        idx + 1,
            "img_path":   img_path or raw_path or "N/A",
            "text":       text,
            "score":      None,
            "error":      None,
            "thumb_b64":  None,
            **token_info(tokenizer, text),
        }

        if not img_path or not os.path.exists(img_path):
            result["error"] = "Image file not found on disk"
            rows.append(result)
            continue

        try:
            pil_img         = Image.open(img_path).convert("RGB")
            result["thumb_b64"] = encode_image_b64(pil_img)
            img_tensor      = transform(pil_img)
            img_tensor_u8   = (img_tensor * 255).to(torch.uint8)
            score           = clip_metric(img_tensor_u8, text)
            result["score"] = round(score.item(), 4)
        except Exception as e:
            result["error"] = str(e)

        rows.append(result)

        if (idx + 1) % 10 == 0:
            done = [r for r in rows if r["score"] is not None]
            avg  = sum(r["score"] for r in done) / len(done) if done else 0
            print(f"  {idx+1}/{len(sample)} processed | valid: {len(done)} | running avg: {avg:.4f}")

    # ── Summary stats ──────────────────────────────────────────────────────────
    scored   = [r for r in rows if r["score"] is not None]
    trunc    = [r for r in rows if r["truncated"]]
    avg_all  = sum(r["score"] for r in scored) / len(scored)  if scored  else 0
    avg_trunc  = sum(r["score"] for r in scored if r["truncated"])     / max(1, sum(1 for r in scored if r["truncated"]))
    avg_notrunc= sum(r["score"] for r in scored if not r["truncated"]) / max(1, sum(1 for r in scored if not r["truncated"]))

    print(f"\n[5/5] Building HTML report → {HTML_OUT}")
    _write_html(rows, scored, trunc, avg_all, avg_trunc, avg_notrunc)
    print(f"Done! Open  {HTML_OUT}  in your browser.")


# ──────────────────────────────────────────────────────────────────────────────
# HTML generation
# ──────────────────────────────────────────────────────────────────────────────

def _write_html(rows, scored, trunc, avg_all, avg_trunc, avg_notrunc):
    trunc_pct = 100 * len(trunc) / len(rows) if rows else 0

    table_rows_html = ""
    for r in rows:
        label, cls = score_label(r["score"]) if r["score"] is not None else ("Error", "error")
        score_str  = f"{r['score']:.4f}" if r["score"] is not None else "—"

        thumb_html = (
            f'<img src="data:image/jpeg;base64,{r["thumb_b64"]}" '
            f'class="thumb" alt="frame {r["idx"]}">'
            if r["thumb_b64"]
            else f'<div class="no-img">No image</div>'
        )

        trunc_badge = (
            '<span class="badge trunc-yes">Yes</span>' if r["truncated"]
            else '<span class="badge trunc-no">No</span>'
        )

        tok_detail = (
            f'{r["tokens_used"]} / {CLIP_TOKEN_LIMIT} used'
            + (f'<br><span class="dropped">−{r["tokens_dropped"]} tokens dropped</span>' if r["truncated"] else "")
        )

        # Show effective text only when truncated (otherwise it's the same as full text)
        effective_cell = (
            f'<details><summary>Show truncated view</summary>'
            f'<p class="effective-text">{_esc(r["effective_text"])}</p></details>'
            if r["truncated"]
            else '<span class="na">Same as full text</span>'
        )

        error_cell = f'<span class="err-msg">{_esc(r["error"])}</span>' if r["error"] else "—"

        table_rows_html += f"""
        <tr class="{cls}">
          <td class="num">{r["idx"]}</td>
          <td class="img-cell">{thumb_html}<br>
              <span class="path" title="{_esc(r['img_path'])}">{_esc(os.path.basename(r['img_path']))}</span>
          </td>
          <td class="text-cell"><div class="scroll-text">{_esc(r['text'])}</div></td>
          <td class="score-cell"><span class="score-badge {cls}">{score_str}</span><br>
              <span class="qlabel">{label}</span></td>
          <td class="tok-cell">{tok_detail}</td>
          <td>{trunc_badge}</td>
          <td class="eff-cell">{effective_cell}</td>
          <td class="err-cell">{error_cell}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CLIP Score Diagnostic Report</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e2e8f0; padding: 24px; }}
  h1   {{ font-size: 1.6rem; margin-bottom: 4px; color: #fff; }}
  .sub {{ color: #94a3b8; font-size: .9rem; margin-bottom: 28px; }}

  /* ── Summary cards ── */
  .cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 32px; }}
  .card  {{ background: #1e2330; border-radius: 12px; padding: 18px 24px; min-width: 160px; flex: 1; }}
  .card .val {{ font-size: 2rem; font-weight: 700; color: #7dd3fc; }}
  .card .lbl {{ font-size: .78rem; color: #64748b; margin-top: 4px; text-transform: uppercase; letter-spacing: .05em; }}
  .card.warn .val {{ color: #fb923c; }}
  .card.good-c .val {{ color: #4ade80; }}
  .card.mod-c  .val {{ color: #facc15; }}

  /* ── Table ── */
  .wrap {{ overflow-x: auto; border-radius: 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .82rem; background: #1e2330; }}
  thead th {{ background: #0f172a; color: #94a3b8; padding: 10px 12px; text-align: left;
               position: sticky; top: 0; z-index: 2; border-bottom: 1px solid #334155; }}
  tbody tr {{ border-bottom: 1px solid #1e293b; }}
  tbody tr:hover {{ background: #273044; }}

  /* row tinting */
  tr.good   {{ border-left: 3px solid #4ade80; }}
  tr.moderate {{ border-left: 3px solid #facc15; }}
  tr.low    {{ border-left: 3px solid #f87171; }}
  tr.error  {{ border-left: 3px solid #94a3b8; opacity: .7; }}

  td {{ padding: 10px 12px; vertical-align: top; }}
  .num {{ color: #64748b; font-size: .75rem; white-space: nowrap; }}

  /* image */
  .img-cell {{ white-space: nowrap; text-align: center; min-width: 180px; }}
  .thumb    {{ width: {THUMB_WIDTH}px; border-radius: 6px; display: block; margin-bottom: 4px; }}
  .no-img   {{ width: {THUMB_WIDTH}px; height: 90px; background: #334155; border-radius: 6px;
                display: flex; align-items: center; justify-content: center; color: #64748b;
                font-size: .7rem; margin-bottom: 4px; }}
  .path     {{ font-size: .68rem; color: #64748b; word-break: break-all; }}

  /* analysis text */
  .text-cell  {{ max-width: 340px; }}
  .scroll-text {{ max-height: 110px; overflow-y: auto; color: #cbd5e1; line-height: 1.5;
                  padding-right: 4px; }}
  .scroll-text::-webkit-scrollbar {{ width: 4px; }}
  .scroll-text::-webkit-scrollbar-thumb {{ background: #334155; border-radius: 2px; }}

  /* score */
  .score-cell  {{ text-align: center; white-space: nowrap; }}
  .score-badge {{ display: inline-block; padding: 4px 10px; border-radius: 20px;
                  font-weight: 700; font-size: 1rem; }}
  .score-badge.good     {{ background: #14532d; color: #4ade80; }}
  .score-badge.moderate {{ background: #422006; color: #facc15; }}
  .score-badge.low      {{ background: #450a0a; color: #f87171; }}
  .score-badge.error    {{ background: #1e293b; color: #94a3b8; }}
  .qlabel {{ font-size: .7rem; color: #64748b; margin-top: 4px; }}

  /* token columns */
  .tok-cell  {{ white-space: nowrap; color: #cbd5e1; }}
  .dropped   {{ color: #f87171; font-size: .78rem; }}
  .badge     {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: .75rem; font-weight: 600; }}
  .trunc-yes {{ background: #450a0a; color: #f87171; }}
  .trunc-no  {{ background: #14532d; color: #4ade80; }}

  /* effective text */
  .eff-cell {{ max-width: 260px; }}
  details summary {{ cursor: pointer; color: #7dd3fc; font-size: .78rem; }}
  .effective-text {{ margin-top: 6px; color: #fbbf24; font-size: .78rem;
                     background: #1c1917; padding: 8px; border-radius: 6px;
                     max-height: 90px; overflow-y: auto; line-height: 1.5; }}
  .na  {{ color: #334155; font-size: .75rem; }}

  /* error */
  .err-msg {{ color: #f87171; font-size: .78rem; }}

  /* filter bar */
  .filters {{ display: flex; gap: 10px; margin-bottom: 18px; flex-wrap: wrap; }}
  .filters button {{ background: #1e2330; border: 1px solid #334155; color: #94a3b8;
                     padding: 6px 14px; border-radius: 20px; cursor: pointer; font-size: .82rem; }}
  .filters button:hover, .filters button.active {{ background: #334155; color: #e2e8f0; }}
</style>
</head>
<body>
<h1>🔍 CLIP Score Diagnostic Report</h1>
<p class="sub">Model: <code>{MODEL_NAME}</code> &nbsp;·&nbsp; Token limit: <strong>{CLIP_TOKEN_LIMIT}</strong> &nbsp;·&nbsp; Frames analysed: <strong>{len(rows)}</strong></p>

<div class="cards">
  <div class="card"><div class="val">{avg_all:.2f}</div><div class="lbl">Overall Avg CLIPScore</div></div>
  <div class="card good-c"><div class="val">{avg_notrunc:.2f}</div><div class="lbl">Avg (no truncation)</div></div>
  <div class="card warn"><div class="val">{avg_trunc:.2f}</div><div class="lbl">Avg (truncated texts)</div></div>
  <div class="card {'warn' if trunc_pct > 40 else ''}"><div class="val">{trunc_pct:.1f}%</div><div class="lbl">Texts truncated</div></div>
  <div class="card"><div class="val">{len(scored)}</div><div class="lbl">Successfully scored</div></div>
  <div class="card"><div class="val">{len(rows)-len(scored)}</div><div class="lbl">Errors / missing</div></div>
</div>

<div class="filters">
  <strong style="align-self:center;color:#64748b">Filter:</strong>
  <button class="active" onclick="filter('all',this)">All ({len(rows)})</button>
  <button onclick="filter('low',this)">🔴 Low (&lt;22)</button>
  <button onclick="filter('moderate',this)">🟡 Moderate (22–30)</button>
  <button onclick="filter('good',this)">🟢 Good (≥30)</button>
  <button onclick="filter('trunc',this)">✂️ Truncated</button>
  <button onclick="filter('error',this)">⚠️ Errors</button>
</div>

<div class="wrap">
<table id="diagTable">
<thead>
  <tr>
    <th>#</th>
    <th>Frame</th>
    <th>Full Analysis Text</th>
    <th>CLIPScore</th>
    <th>Tokens Used</th>
    <th>Truncated?</th>
    <th>Text CLIP Actually Sees</th>
    <th>Error</th>
  </tr>
</thead>
<tbody>
{table_rows_html}
</tbody>
</table>
</div>

<script>
function filter(kind, btn) {{
  document.querySelectorAll('.filters button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('#diagTable tbody tr').forEach(tr => {{
    if (kind === 'all')   {{ tr.style.display = ''; return; }}
    if (kind === 'trunc') {{ tr.style.display = tr.querySelector('.trunc-yes') ? '' : 'none'; return; }}
    tr.style.display = tr.classList.contains(kind) ? '' : 'none';
  }});
}}
</script>
</body>
</html>"""

    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write(html)


def _esc(s: str) -> str:
    """Basic HTML escaping."""
    if not s:
        return ""
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;"))


if __name__ == "__main__":
    main()