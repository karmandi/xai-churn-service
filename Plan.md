# Project Master Plan: Autonomous Telco Churn XAI & Remediation Service

## 1. Engineering Philosophy & Portfolio Goal
This repository is built as a **production-grade AI engineering system**, not an exploratory notebook. Every phase scales incrementally from a deterministic ML baseline into an autonomous, evaluated, and resource-profiled AI service.

### Core Architectural Principles
- **Separation of Concerns:** Strict modular layout (`src/data`, `src/models`, `src/api`, `src/agent`, `src/evals`).
- **Deterministic Guardrails:** All inputs, outputs, and agent tool calls are validated via Pydantic v2 schemas.
- **Fail-Safe Degradation:** External tools and LLM calls enforce strict timeouts, retries, and graceful fallbacks.
- **Evidence-Driven Iteration:** Every phase is verified with automated tests (`pytest`), latency metrics, and evaluation gates.

---

## 2. Current State (Completed)
- **Dataset:** IBM Telco Customer Churn (`data/churn.csv`, 7,043 rows).
- **Features (Scoped Baseline):** `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `PaymentMethod`.
- **Completed Modules:**
  - `src/data/eda.py`: Missing value & segment churn analysis.
  - `src/models/train.py`: `ColumnTransformer` + tuned `RandomForestClassifier(n_estimators=200, max_depth=6)` saved to `artifacts/model.joblib`.
  - `src/models/compare.py`: 5-fold stratified `GridSearchCV` comparing `RandomForestClassifier` (`0.8438`), `HistGradientBoostingClassifier` (`0.8437`), and `LogisticRegression` (`0.8388`).
  - `src/models/explainer.py`: `ChurnExplainer` computing churn probability, risk tier (`Low`/`Medium`/`High`), and top 3 `shap.TreeExplainer` attributions.
  - `src/models/evaluate.py`: Verified leak-free holdout `ROC-AUC: 0.8416` (`80%` accuracy) and SHAP directional consistency.

---

## 3. Phased Scaling Roadmap

### Phase 1: Minimal XAI Prediction API (Current Active Phase)
- **Goal:** Serve real-time predictions and SHAP attributions via a tested REST API.
- **Tasks:**
  1. `src/api/schemas.py`: Define `CustomerPayload` and `PredictionResponse` (Pydantic v2).
  2. `src/api/main.py`: Expose `GET /health` and `POST /predict-explain`.
  3. `tests/test_api.py`: Verify HTTP 200 valid responses and HTTP 422 schema rejections via `TestClient`.

### Phase 2: Agent Harness (Autonomous Remediation Loop)
- **Goal:** When churn risk is `High` (>= 0.70), trigger a bounded controller loop that investigates root causes and proposes/executes safe retention actions.
- **Components to Build (`src/agent/`):**
  1. **Controller Loop (`harness.py`):** Max 3 iterations; decides tool calls, parses structured outputs, and knows when to stop.
  2. **Tool Layer (`tools.py`):** Typed tools (`get_shap_breakdown`, `check_feature_drift`, `apply_retention_perk`) with Pydantic input schemas, 2-second timeouts, and retry wrappers.
  3. **Context Manager (`context.py`):** Tracks token budget and summarizes prior turns to prevent context overflow.
  4. **Sandbox & Tracing (`tracing.py`):** Enforces read/write permission boundaries against prompt injection and logs every step's latency, token count, and tool status.
  5. **Chaos Testing (`tests/test_chaos.py`):** Intentionally break the harness with hung tools, infinite loops, and malformed outputs to verify recovery.

### Phase 3: End-to-End Evals & Regression Gate
- **Goal:** Grade system reasoning, explanation quality, and resilience—not just model accuracy.
- **Components to Build (`src/evals/`):**
  1. **Eval Runner (`runner.py`):** Runs a 100-case golden dataset through the full pipeline and outputs a structured report.
  2. **Multi-Layer Graders (`graders.py`):**
     - *Deterministic Graders:* Verify SHAP additivity and valid tool selection sequences (grading the trajectory, not just final text).
     - *LLM-as-a-Judge:* Evaluates clarity and business accuracy of retention explanations, calibrated against 50+ human-labeled examples.
  3. **Edge-Case & Adversarial Suite:** Tests null/outlier inputs, tool timeouts, and prompt injection payloads.
  4. **CI/CD Regression Gate (`.github/workflows/ci.yml`):** Blocks merges if `pass@k`, `p95` latency, or cost-per-request regress.

### Phase 4: Narrow Fine-Tuning & vLLM Serving Metrics
- **Goal:** Replace slow/expensive general LLM calls with a compact, domain-tuned explainer model and profile hardware efficiency.
- **Tasks:**
  1. Benchmark a prompting-only baseline on a small open model (e.g., Llama 3 8B) for translating SHAP dicts into 2-sentence retention strategies.
  2. Fine-tune using **QLoRA** on ~500 `(SHAP_json, retention_summary)` pairs.
  3. Serve via **vLLM** and measure **TTFT**, **inter-token latency**, **p95 latency**, and **goodput**.
  4. Document analytical KV-cache memory math (`2 * layers * kv_heads * head_dim * bytes`) and max concurrency limits in `README.md`.
