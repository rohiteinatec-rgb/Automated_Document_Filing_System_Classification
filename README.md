# 🏛️ Automated Document Filing System (ADFS)

**Status:** `Production-Ready (Pilot Phase)` | **Architecture:** `Zero-Egress / On-Premises`

ADFS is an asynchronous, AI-driven document orchestration pipeline designed to automatically extract, classify, and archive corporate documents (invoices, contracts, budgets) with absolute data privacy.

## 🔒 Core Architectural Principles

1. **Zero-Egress Data Security:** 100% of LLM processing is handled locally via Ollama (Qwen3). No proprietary corporate data or PII ever leaves the internal network.
2. **Deterministic Guardrails:** The system does not rely solely on LLM probabilistic outputs. Mathematical validators (e.g., Tax ID checksums, binding clause regex) act as hard gates before any file is routed.
3. **Graceful Degradation:** Features a built-in memory quarantine to prevent AI hallucination loops, exponential database backoffs, and strict disk-space circuit breakers.

## 🛠️ Tech Stack
* **Engine:** Python 3.10+, FastAPI, AsyncIO
* **AI/ML:** Ollama (Qwen3-8B), RapidOCR, ChromaDB
* **Data Layer:** PostgreSQL (psycopg2 ThreadedConnectionPool)
* **Telemetry:** Custom JSONL Event Logging & HTML Metric Aggregation

## 🚀 Quick Start (Local Pilot)

### 1. Environment Setup
Create a `.env` file in the root directory:
```env
OLLAMA_BASE_URL=http://localhost:11434
DATABASE_URL=postgresql://user:password@localhost:5432/adfs
OUTPUT_ROOT=./output
