import time
import shutil
import requests
import json
from pathlib import Path
from typing import List, Dict
from config import Config
from errors import AlertManager

class ObservabilityManager:
    def __init__(self, metrics_log: str = "adfs_metrics.jsonl"):
        self.metrics_log = metrics_log
        self.metrics: List[Dict] = []

        # Ensure the log file exists
        if not Path(self.metrics_log).exists():
            Path(self.metrics_log).touch()

    def check_health(self) -> bool:
        """Runs pre-flight checks before starting a massive batch queue."""
        print("\n  [Observability] 🩺 Running pre-flight system health checks...")
        is_healthy = True

        # 1. Ollama Health Check
        try:
            # We hit the base URL to ensure the container/app is alive
            r = requests.get(Config.OLLAMA_BASE_URL, timeout=3)
            if r.status_code == 200:
                print("    ✅ Ollama API   : Online & Responsive")
            else:
                AlertManager.send_alert("HEALTH_CHECK", f"Ollama returned HTTP {r.status_code}", "CRITICAL")
                is_healthy = False
        except requests.exceptions.RequestException:
            AlertManager.send_alert("HEALTH_CHECK", "Ollama API unreachable. Is the app running?", "CRITICAL")
            is_healthy = False

        # 2. Output Disk Space Check (Require at least 1GB)
        try:
            output_dir = Path(Config.OUTPUT_DIR)
            output_dir.mkdir(parents=True, exist_ok=True)
            total, used, free = shutil.disk_usage(output_dir)
            free_gb = free // (2**30)

            if free_gb < 1:
                AlertManager.send_alert("HEALTH_CHECK", f"Low disk space on output drive ({free_gb}GB left)", "CRITICAL")
                is_healthy = False
            else:
                print(f"    ✅ Disk Space   : {free_gb} GB Free")
        except Exception as e:
            AlertManager.send_alert("HEALTH_CHECK", f"Cannot access output directory: {e}", "CRITICAL")
            is_healthy = False

        return is_healthy

    def record_metric(self, name: str, value: float, tags: dict = None):
        """Records a metric in memory and appends to the local JSONL log."""
        tags = tags or {}
        metric_data = {
            "timestamp": time.time(),
            "metric": name,
            "value": value,
            "tags": tags
        }

        # Keep in memory for the end-of-batch summary
        self.metrics.append(metric_data)

        # Write to log for long-term drift detection / analysis
        with open(self.metrics_log, 'a') as f:
            f.write(json.dumps(metric_data) + '\n')

    def print_batch_sli_summary(self):
        """Calculates p95 and averages for the CLI output."""
        if not self.metrics:
            return

        latencies = [m["value"] for m in self.metrics if m["metric"] == "classifier.latency_ms"]

        if latencies:
            latencies.sort()
            avg_lat = sum(latencies) / len(latencies)
            # Calculate 95th percentile
            p95_idx = int(len(latencies) * 0.95)
            p95_lat = latencies[p95_idx] if p95_idx < len(latencies) else latencies[-1]

            print(f"\n  📊 OBSERVABILITY SLI REPORT:")
            print(f"    Avg Model Latency : {avg_lat:.2f} ms")
            print(f"    p95 Model Latency : {p95_lat:.2f} ms")

            if p95_lat > 60000: # 60 seconds
                AlertManager.send_alert("SLI_BREACH", f"p95 latency exceeded 60s target ({p95_lat/1000:.1f}s)", "WARNING")