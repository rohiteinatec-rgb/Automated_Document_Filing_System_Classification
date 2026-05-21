import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
from config import Config

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

    def generate_html_dashboard(self, output_file: str = "dashboard.html"):
        """Generate a clean, local-first HTML dashboard."""
        summary = self.get_performance_summary()
        output_path = os.path.join(Config.BASE_DIR, output_file)

        html = f"""
        <html>
        <head>
            <title>ADFS Metrics Dashboard</title>
            <style>
                body {{ font-family: 'Segoe UI', Arial; margin: 40px; background: #f4f7f6; }}
                .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #2c3e50; color: white; }}
                .status-ok {{ color: green; font-weight: bold; }}
                .status-warn {{ color: orange; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="card">
                <h1>🚀 ADFS Performance Dashboard</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Files Processed (24h)</td><td>{summary.get('total_files', 0)}</td></tr>
                    <tr><td>Average Latency</td><td>{summary.get('avg_latency_ms', 0):.0f} ms</td></tr>
                    <tr><td>p95 Latency (Tail Performance)</td><td>{summary.get('p95_latency_ms', 0):.0f} ms</td></tr>
                    <tr>
                        <td>Slow Files (>120s)</td>
                        <td class="{'status-warn' if summary.get('slow_files_count', 0) > 0 else 'status-ok'}">
                            {summary.get('slow_files_count', 0)}
                        </td>
                    </tr>
                </table>
            </div>
        </body>
        </html>
        """
        with open(output_path, "w") as f:
            f.write(html)
        print(f"  [Metrics] Dashboard generated: {output_path}")