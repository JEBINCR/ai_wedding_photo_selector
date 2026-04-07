"""
Generates HTML and CSV reports of the analysis results.
"""

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ReportGenerator:
    def generate(self, all_results: list, selected: list, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        self._write_csv(all_results, output_dir / "all_scores.csv")
        self._write_html(all_results, selected, output_dir / "report.html")
        logger.info(f"Reports saved to {output_dir}")

    # ------------------------------------------------------------------
    def _write_csv(self, results: list, path: Path):
        fields = [
            "rank", "filename", "score", "face_count",
            "blur_score", "is_blurry", "smile_score",
            "eyes_open", "eye_open_ratio", "error"
        ]
        sorted_r = sorted(results, key=lambda r: r["score"], reverse=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for rank, row in enumerate(sorted_r, 1):
                writer.writerow({
                    "rank": rank,
                    "filename": row["filename"],
                    "score": row["score"],
                    "face_count": row["face_count"],
                    "blur_score": round(row["blur_score"], 1),
                    "is_blurry": row["is_blurry"],
                    "smile_score": round(row["smile_score"], 3),
                    "eyes_open": row["eyes_open"],
                    "eye_open_ratio": round(row["eye_open_ratio"], 3),
                    "error": row.get("error") or "",
                })

    def _write_html(self, all_results: list, selected: list, path: Path):
        selected_names = {r["filename"] for r in selected}
        sorted_r = sorted(all_results, key=lambda r: r["score"], reverse=True)

        rows_html = ""
        for rank, r in enumerate(sorted_r, 1):
            highlight = 'class="selected"' if r["filename"] in selected_names else ""
            error_badge = f'<span class="badge error">ERR</span>' if r.get("error") else ""
            blur_badge = '<span class="badge blurry">BLUR</span>' if r["is_blurry"] else ""
            rows_html += f"""
            <tr {highlight}>
              <td>{rank}</td>
              <td>{r['filename']}{error_badge}{blur_badge}</td>
              <td><strong>{r['score']}</strong></td>
              <td>{r['face_count']}</td>
              <td>{r['blur_score']:.0f}</td>
              <td>{'✅' if not r['is_blurry'] else '❌'}</td>
              <td>{r['smile_score']:.2f}</td>
              <td>{'✅' if r['eyes_open'] else '❌'}</td>
              <td>{r['eye_open_ratio']:.2f}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Wedding Photo Selector Report</title>
<style>
  body {{ font-family: Arial, sans-serif; padding: 20px; background: #fafafa; }}
  h1 {{ color: #b5477a; }}
  .summary {{ display: flex; gap: 24px; margin: 16px 0; }}
  .stat {{ background: white; border-radius: 8px; padding: 16px 24px;
           box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; }}
  .stat .val {{ font-size: 2em; font-weight: bold; color: #b5477a; }}
  table {{ width: 100%; border-collapse: collapse; background: white;
           box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-radius: 8px; overflow: hidden; }}
  th {{ background: #b5477a; color: white; padding: 10px 14px; text-align: left; }}
  td {{ padding: 8px 14px; border-bottom: 1px solid #eee; }}
  tr.selected {{ background: #fff0f7; }}
  tr:hover {{ background: #f5e6f0; }}
  .badge {{ font-size: 0.7em; padding: 2px 6px; border-radius: 4px;
            margin-left: 6px; vertical-align: middle; }}
  .badge.error {{ background: #ffcccc; color: #c00; }}
  .badge.blurry {{ background: #ffe5cc; color: #a60; }}
</style>
</head>
<body>
<h1>💍 Wedding Photo Selector — Analysis Report</h1>
<div class="summary">
  <div class="stat"><div class="val">{len(all_results)}</div>Total Analysed</div>
  <div class="stat"><div class="val">{len(selected)}</div>Selected</div>
  <div class="stat"><div class="val">{sum(1 for r in all_results if not r['is_blurry'])}</div>Sharp Images</div>
  <div class="stat"><div class="val">{sum(1 for r in all_results if r['face_count']>0)}</div>With Faces</div>
  <div class="stat"><div class="val">{sum(1 for r in all_results if r.get('eyes_open'))}</div>Eyes Open</div>
</div>
<p>Rows highlighted in pink are in the top selection. Sorted by composite score.</p>
<table>
  <thead>
    <tr>
      <th>Rank</th><th>Filename</th><th>Score</th><th>Faces</th>
      <th>Blur ▲</th><th>Sharp?</th><th>Smile</th><th>Eyes Open?</th><th>Eye Ratio</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
</body>
</html>"""
        path.write_text(html, encoding="utf-8")
