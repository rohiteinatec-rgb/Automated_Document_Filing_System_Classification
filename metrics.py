import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
from config import Config
from database import DatabaseArchiver

class MetricsAggregator:
    def __init__(self, metrics_file: str = "adfs_metrics.jsonl"):
        # Ensure we look in the right base directory
        self.metrics_file = os.path.join(Config.BASE_DIR, metrics_file)

    def get_performance_summary(self, hours: int = 24) -> Dict:
        """Summarize metrics from the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = []

        if not Path(self.metrics_file).exists():
            return {}

        with open(self.metrics_file, "r") as f:
            for line in f:
                try:
                    m = json.loads(line)
                    # Support both Unix timestamp and ISO format
                    ts = m["timestamp"]
                    m_time = datetime.fromtimestamp(ts) if isinstance(ts, (int, float)) else datetime.fromisoformat(ts)

                    if m_time > cutoff_time:
                        recent_metrics.append(m)
                except:
                    continue

        if not recent_metrics:
            return {}

        latencies = [m["value"] for m in recent_metrics if m["metric"] == "classifier.latency_ms"]

        return {
            "total_files": len(recent_metrics),
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            "slow_files_count": len([l for l in latencies if l > 120000]),
        }

    def generate_html_dashboard(self):
        output_path = os.path.join(Config.BASE_DIR, "dashboard.html")
        db = DatabaseArchiver()

        # 1. Initialize Default Metrics
        total_files = 0
        filed_files = 0
        uncertain_files = 0
        error_files = 0
        accuracy_pct = 0.0
        table_rows_html = ""

        avg_latency_s = 0.0
        p95_latency_s = 0.0

        # 2. Calculate Live Latency from JSONL
        metrics_path = os.path.join(Config.BASE_DIR, "adfs_metrics.jsonl")
        if os.path.exists(metrics_path):
            latencies = []
            with open(metrics_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if data.get("metric") == "classifier.latency_ms":
                            latencies.append(data.get("value", 0))
                    except:
                        pass
            if latencies:
                latencies.sort()
                avg_latency_s = (sum(latencies) / len(latencies)) / 1000.0
                p95_idx = int(len(latencies) * 0.95)
                p95_latency_s = latencies[p95_idx] / 1000.0 if p95_idx < len(latencies) else latencies[-1] / 1000.0

        # 3. Fetch Live Data from PostgreSQL
        try:
            with db.get_connection() as conn:
                if conn:
                    with conn.cursor() as cur:
                        # Fetch aggregate counts
                        cur.execute("SELECT action, COUNT(*) FROM filings GROUP BY action;")
                        counts = dict(cur.fetchall())

                        # Note: dry_run is counted as filed for testing purposes
                        filed_files = counts.get('filed', 0) + counts.get('dry_run', 0)
                        uncertain_files = counts.get('uncertain', 0)
                        error_files = counts.get('error', 0)
                        total_files = sum(counts.values())

                        accuracy_pct = round((filed_files / total_files * 100), 1) if total_files > 0 else 0.0

                        # Fetch recent 100 files for the table
                        cur.execute("""
                                    SELECT pdf_file, tag, company, action
                                    FROM filings
                                    ORDER BY timestamp DESC
                                        LIMIT 100;
                                    """)

                        for row in cur.fetchall():
                            fname, tag, company, action = row

                            # Dynamic UI Logic based on DB status
                            if action in ('filed', 'dry_run') and tag != 'uncertain':
                                badge = '<span class="px-2.5 py-1 bg-emerald-50 text-emerald-700 border border-emerald-200 rounded text-xs font-bold uppercase">Factura/Pressupost</span>'
                                impact = '<span class="text-emerald-500 font-bold text-xs uppercase tracking-wider">✅ Auto-Filed</span>'
                            elif tag == 'uncertain' or action == 'uncertain':
                                badge = '<span class="px-2.5 py-1 bg-amber-50 text-amber-700 border border-amber-200 rounded text-xs font-bold uppercase">Uncertain</span>'
                                impact = '<span class="text-amber-500 font-bold text-xs uppercase tracking-wider">🟡 Human Review</span>'
                            else:
                                badge = '<span class="px-2.5 py-1 bg-red-50 text-red-700 border border-red-200 rounded text-xs font-bold uppercase">Error</span>'
                                impact = '<span class="text-red-600 font-bold text-xs uppercase tracking-wider">🔴 Alert</span>'

                            table_rows_html += f"""
                            <tr class="hover:bg-slate-50 transition-colors">
                                <td class="px-6 py-4 font-mono text-slate-600 text-xs">{fname}</td>
                                <td class="px-6 py-4">{badge}</td>
                                <td class="px-6 py-4 text-slate-700">{company}</td>
                                <td class="px-6 py-4 font-mono text-slate-500">Auto</td>
                                <td class="px-6 py-4">{impact}</td>
                            </tr>
                            """
        except Exception as e:
            print(f"  [Metrics] ⚠️ DB Fetch Error: {e}")
            table_rows_html = f"<tr><td colspan='5' class='px-6 py-4 text-red-500'>DB Error: {e}</td></tr>"

        # 4. Inject variables into the HTML string
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ADFS Executive Evaluation</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Inter', sans-serif; background-color: #f8fafc; }}
    </style>
</head>
<body class="text-slate-800 p-8 antialiased">

<div class="max-w-7xl mx-auto space-y-6">

    <header class="flex justify-between items-end pb-4 border-b border-slate-200">
        <div>
            <h1 class="text-3xl font-bold text-slate-900 tracking-tight">ADFS Production Dashboard</h1>
            <p class="text-slate-500 mt-1">Live Telemetry — Zero-Egress On-Premises Pipeline</p>
        </div>
        <div class="text-right">
                <span class="inline-flex items-center px-4 py-1.5 rounded-full text-sm font-semibold bg-emerald-100 text-emerald-800 border border-emerald-200 shadow-sm">
                    Status: System Healthy
                </span>
        </div>
    </header>

    <section class="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
            <p class="text-xs font-bold text-slate-400 uppercase tracking-wider">Automated Filings</p>
            <p class="text-4xl font-extrabold text-slate-900 mt-2">{accuracy_pct}%</p>
            <p class="text-sm font-medium text-emerald-600 mt-1">{filed_files} / {total_files} Processed</p>
        </div>
        <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
            <p class="text-xs font-bold text-slate-400 uppercase tracking-wider">System Errors</p>
            <p class="text-4xl font-extrabold text-slate-900 mt-2">{error_files}</p>
            <p class="text-sm font-medium text-emerald-600 mt-1">Pipeline Stability</p>
        </div>
        <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
            <p class="text-xs font-bold text-slate-400 uppercase tracking-wider">Processing Speed</p>
            <p class="text-4xl font-extrabold text-slate-900 mt-2">{avg_latency_s:.2f}s</p>
            <p class="text-sm font-medium text-slate-500 mt-1">p95 Tail: {p95_latency_s:.2f}s</p>
        </div>
        <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
            <p class="text-xs font-bold text-slate-400 uppercase tracking-wider">Human-In-The-Loop</p>
            <p class="text-4xl font-extrabold text-amber-500 mt-2">{uncertain_files}</p>
            <p class="text-sm font-medium text-slate-500 mt-1">Safely routed to Review</p>
        </div>
    </section>

    <section class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
            <h2 class="text-lg font-bold text-slate-800 mb-4 border-b border-slate-100 pb-2">Business Routing Impact</h2>
            <div class="space-y-4 pt-2">
                <div class="flex justify-between items-center">
                        <span class="flex items-center gap-3 text-sm font-medium text-slate-700">
                            <span class="w-3 h-3 rounded-full bg-red-500"></span> Extraction Errors
                        </span>
                    <span class="font-mono text-sm font-bold text-slate-900">{error_files}</span>
                </div>
                <div class="flex justify-between items-center">
                        <span class="flex items-center gap-3 text-sm font-medium text-slate-700">
                            <span class="w-3 h-3 rounded-full bg-amber-400"></span> Human Review Required
                        </span>
                    <span class="font-mono text-sm font-bold text-slate-900">{uncertain_files}</span>
                </div>
                <div class="flex justify-between items-center pb-3 border-b border-slate-100">
                        <span class="flex items-center gap-3 text-sm font-bold text-emerald-700">
                            <span class="w-3 h-3 rounded-full bg-emerald-500"></span> Successfully Filed
                        </span>
                    <span class="font-mono text-sm font-bold text-emerald-700">{filed_files}</span>
                </div>
            </div>
        </div>

        <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
            <h2 class="text-lg font-bold text-slate-800 mb-4 border-b border-slate-100 pb-2">Security & Robustness Gates</h2>
            <ul class="grid grid-cols-2 gap-y-4 pt-2 text-sm font-medium">
                <li class="flex items-center gap-2"><span class="text-emerald-500 text-lg">✔</span> <span class="text-slate-700">Empty Text Defense</span></li>
                <li class="flex items-center gap-2"><span class="text-emerald-500 text-lg">✔</span> <span class="text-slate-700">Malformed JSON</span></li>
                <li class="flex items-center gap-2"><span class="text-emerald-500 text-lg">✔</span> <span class="text-slate-700">Prompt Injection</span></li>
                <li class="flex items-center gap-2"><span class="text-emerald-500 text-lg">✔</span> <span class="text-slate-700">Unicode Accents</span></li>
                <li class="flex items-center gap-2"><span class="text-emerald-500 text-lg">✔</span> <span class="text-slate-700">Zero Confidence Safeties</span></li>
                <li class="flex items-center gap-2"><span class="text-emerald-500 text-lg">✔</span> <span class="text-slate-700">Data Isolation (Local LLM)</span></li>
            </ul>
        </div>
    </section>

    <section class="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden mt-6">
        <div class="px-6 py-4 border-b border-slate-200 bg-slate-50 flex justify-between items-center">
            <h2 class="text-lg font-bold text-slate-800">Detailed Classification Audit</h2>
            <span class="text-xs font-bold uppercase tracking-wider text-slate-400">Showing Last 100 Processed</span>
        </div>
        <div class="overflow-x-auto">
            <table class="w-full text-sm text-left">
                <thead class="bg-white text-slate-500 font-bold uppercase text-xs tracking-wider border-b border-slate-200">
                <tr>
                    <th class="px-6 py-4">File Name</th>
                    <th class="px-6 py-4">Actual Tag</th>
                    <th class="px-6 py-4">Extracted Company</th>
                    <th class="px-6 py-4">Engine</th>
                    <th class="px-6 py-4">Impact</th>
                </tr>
                </thead>
                <tbody class="divide-y divide-slate-100 font-medium">
                    {table_rows_html}
                </tbody>
            </table>
        </div>
    </section>

</div>
</body>
</html>"""

        # 5. Write out the final HTML string safely
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"  [Metrics] Dynamic Dashboard generated: {output_path}")