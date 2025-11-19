## MCP Playbook for GPT-5.1-codex

---
trigger: always_on
---

**CRITICAL: READ THIS FIRST - MANDATORY MEMORY PROTOCOL**

You are an AI coding assistant working in the LaTeXify repository with access to CORE Memory and local memory MCPs – persistent knowledge systems that maintain project context, learnings, and continuity across all coding sessions.

##  MANDATORY STARTUP SEQUENCE - DO NOT SKIP 

**BEFORE RESPONDING TO ANY USER MESSAGE, YOU MUST EXECUTE THESE TOOLS IN ORDER:**

### STEP 1 (REQUIRED): Search for Relevant Context

EXECUTE THIS TOOL FIRST:
`memory_search` (via the CORE/Memory MCP layer configured for this environment)

You must search for:

- Previous discussions about the current topic
- Related project decisions and implementations
- User preferences and work patterns
- Similar problems and their solutions

**Additional search triggers:**

- User mentions "previously", "before", "last time", or "we discussed"
- User references past work or project history
- Working on this project (LaTeXify) in any capacity
- User asks about preferences, patterns, or past decisions
- Starting work on any feature or bug that might have history

**How to search effectively:**

- Write complete semantic queries, NOT keyword fragments
- Good: `"LaTeXify release pipeline architecture and GPU OCR scheduling decisions"`
- Good: `"User preferences for LaTeX aesthetic (booktabs, amsmath, hyperref) in LaTeXify outputs"`
- Bad: `"latexify pipeline"`
- Ask: "What context am I missing that would help?"
- Consider: "What has the user told me before that I should remember?"

### Query Patterns for Memory Search

**Entity-Centric Queries** (Best for graph search):

-  GOOD: `"LaTeXify project OCR backend selection and VRAM tradeoffs"`
-  GOOD: `"User's preferences for LaTeX package usage and document aesthetics"`
-  BAD: `"ocr prefs"`
- Format: `[Person/Project] + [relationship/attribute] + [context]`

**Multi-Entity Relationship Queries** (Excellent for episode graph):

-  GOOD: `"relationship between layout segmentation and retrieval quality in LaTeXify"`
-  GOOD: `"discussions connecting InternVL, Florence, and Nougat in the OCR cascade"`
-  BAD: `"internvl florence nougat"`
- Format: `[Entity1] + [relationship type] + [Entity2] + [context]`

**Semantic Question Queries** (Good for vector search):

-  GOOD: `"What caused previous LaTeXify runs to OOM on a single 32GB GPU and how was it mitigated?"`
-  GOOD: `"What requirements do LaTeXify's agents impose on GraphState fields and metadata contracts?"`
-  BAD: `"oom issues"`
- Format: Complete natural questions with full context

**Concept Exploration Queries** (Good for BFS traversal):

-  GOOD: `"concepts and ideas related to semantic chunking and SCAN-style layout segmentation in LaTeXify"`
-  GOOD: `"topics connected to reward shaping and aesthetic scoring in LaTeXify's reward schema"`
-  BAD: `"semantic relevance concepts"`
- Format: `[concept] + related/connected + [domain/context]`

**Temporal Queries** (Good for recent work):

-  GOOD: `"recent changes to OCR scheduling, RAG caching, and quality gating logic in LaTeXify"`
-  GOOD: `"latest discussions about LangGraph integration and Experimental Agent Stack orchestration for LaTeXify"`
-  BAD: `"recent search changes"`
- Format: `[temporal marker] + [specific topic] + [additional context]`

##  MANDATORY SHUTDOWN SEQUENCE - DO NOT SKIP 

**AFTER FULLY RESPONDING TO THE USER, YOU MUST EXECUTE THIS TOOL:**

### FINAL STEP (REQUIRED): Store Conversation Memory

EXECUTE THIS TOOL LAST:
`memory_ingest`

Include the `spaceId` parameter using the ID from your initial `memory_get_space` call.

 **THIS IS NON-NEGOTIABLE** - You must ALWAYS store conversation context as your final action.

**What to capture in the `message` parameter:**

From User:

- Specific question, request, or problem statement
- Project context and situation provided
- What they're trying to accomplish
- Technical challenges or constraints mentioned

From Assistant:

- Detailed explanation of solution/approach taken
- Step-by-step processes and methodologies
- Technical concepts and principles explained
- Reasoning behind recommendations and decisions
- Alternative approaches discussed
- Problem-solving methodologies applied

**Include in storage:**

- All conceptual explanations and theory
- Technical discussions and analysis
- Problem-solving approaches and reasoning
- Decision rationale and trade-offs
- Implementation strategies (described conceptually)
- Learning insights and patterns

**Exclude from storage:**

- Code blocks and code snippets
- File contents or file listings
- Command examples or CLI commands
- Raw data or logs

**Quality check before storing:**

- Can someone quickly understand project context from memory alone?
- Would this information help provide better assistance in future sessions?
- Does stored context capture key decisions and reasoning?

---

## Refactor & Architecture Cleanup Workflow

When I say things like “run the refactor plan”, “start the architecture cleanup”, or “follow the MCP cleanup workflow”, you MUST enter **Refactor Mode** and follow this loop exactly, step by step.

### Refactor Mode: High-Level Goal

- Populate and refresh the LaTeXify architecture graph in the **memory** MCP.
- Clean up redundant modules and messy folder layout in `LaTeXify/` without breaking public contracts or artifacts.

### Tools to Prefer

- **memory** MCP for graph entities/relations and session summaries.
- **git** MCP for status, diffs, logs, and scoping changes.
- **sequential_thinking** MCP to generate and refine stepwise plans.
- **web_search** + **context7** for external docs / error diagnosis.
- **github** MCP only when I explicitly ask to open issues/PRs.

### Step-Cycling Loop (DO NOT SKIP STEPS)

1. **Bootstrap / Step 0 – Context & Graph Sync**
   - Use **memory** MCP to search for existing LaTeXify architecture entries.
   - If they are missing or clearly stale, scan `LaTeXify/` (especially `release/` and `release/agents/`) and reconstruct:
     - Subsystems
     - Modules
     - Files
     - Artifacts/contracts
   - Write or update entities/relations in **memory** so the graph matches the current repo state.

2. **Step 1 – Plan with sequential_thinking**
   - Call **sequential_thinking** MCP to produce a numbered plan (≤ 10 steps) that:
     - Names specific files/modules to touch.
     - Identifies which tests/smoke runs to use per step.
     - Identifies which memory entities/relations will need updates.
   - Present the plan as a clear, numbered list.
   - **STOP here and ask me explicitly:**  
     “Here is the plan. Do you want me to execute step 1 now?”

3. **Step 2+ – Execute a Single Plan Step (Per Cycle)**
   For each step `N` in the plan, when I say “yes”, “go ahead with step N”, or similar:

   1. **Re-state the step** in your own words so it’s clear what will happen.
   2. Use **git** MCP to:
      - Show `git status`.
      - Summarize any existing diffs under `LaTeXify/`, especially `release/` and `release/agents/`.
   3. **Propose the edits before applying them**:
      - Describe what changes you intend to make (files, functions, contracts).
      - Ask: “Confirm that I should apply these edits for step N?”  
        Do not write to disk until I confirm.
   4. After I confirm:
      - Apply the edits.
      - Use **git** MCP to show the diff for this step (scoped to the relevant files).
      - Run the agreed tests/smoke checks (describe commands and summarize results).
      - If tests fail, STOP the loop and switch into debugging mode instead of continuing.

4. **Step 3 – Graph & Memory Update**
   After each completed step (tests pass):

   - Use **memory** MCP to:
     - Add/update entities and relations for any new/renamed/moved modules/files.
     - Remove entities/relations for any deleted modules/files.
   - Add or update a short summary observation such as:
     - `LaTeXify: Refactor Step N – <short label>`
   - Briefly report what changed in the architecture graph.

5. **Step 4 – Human Review Gate**
   - Present:
     - A plain-language summary of the code changes.
     - The key diff chunks.
     - The updated architecture view for the affected subsystems.
   - Then ask explicitly:  
     “Do you approve this step and want me to continue to step N+1, or should we revise this step?”

   - If I say “revise” or “adjust”, stay on the same step:
     - Refine the code and/or graph updates until I approve.
     - Only then offer to proceed to the next step.

6. **Step 5 – Session Wrap-Up**
   When we have executed all approved steps or I say “stop”:

   - Use **git** MCP to summarize all diffs made in this session.
   - Use **memory** MCP to write a final session summary entry, including:
     - What structural changes were made.
     - Which subsystems/modules were simplified.
     - Any TODOs or follow-ups for future sessions.
   - Present:
     - A diff summary of structural changes.
     - A snapshot/description of the current architecture graph.
     - A list of remaining opportunities for cleanup.

### Approval & Safety Expectations

- Never auto-advance from one plan step to the next without explicit textual confirmation from me.
- Never delete or move modules/files outside `LaTeXify/` unless I specifically ask.
- Prefer small, reversible steps; rely on **git** MCP and tests to guard each step.

---

## 0. Session Bootstrap (Every Run)

In addition to the Mandatory Memory Protocol above, use this bootstrap routine:

1. **Run the memory protocol first**  
   - Execute `memory_search` as described above *before* any other tools.  
   - If relevant context is found, summarize it mentally and treat it as constraints.

2. **Scope check via `git` MCP**  
   - Call `git` to get `status` and current branch, focusing on changes under `release/` and `release/agents/`.  
   - If there are unrelated modifications outside these areas, treat them as out of scope unless the user says otherwise. :contentReference[oaicite:3]{index=3}  

3. **Plan with `sequential_thinking` for non-trivial tasks**  
   - For any change beyond a trivial fix, call `sequential_thinking` to:
     - Produce ≤10 steps,
     - Reference specific files/modules,
     - Include validation steps (pytest, smoke runs, metric checks). :contentReference[oaicite:4]{index=4}  

4. **Stay inside the harness rules**  
   - When searching code, prefer `rg`/`rg --files` over `grep`.  
   - Keep shell commands scoped to this repo and avoid destructive commands unless explicitly requested. :contentReference[oaicite:5]{index=5}  

---

## 1. Memory & CORE Memory — Long-Term Project Knowledge

Use `memory` and `core_memory` to avoid rediscovering decisions.

**READ from memory when:**

- Starting any substantial change to:
  - OCR cascade (Florence → InternVL → Nougat → pytesseract). :contentReference[oaicite:6]{index=6}  
  - Layout segmentation and RAG index structure.
  - Agent contracts and GraphState fields in `release/agents/`. :contentReference[oaicite:7]{index=7}  
- Revisiting a feature/bug that has likely been discussed before.

**WRITE to memory when:**

- You change:
  - Pipeline architecture or data contracts,
  - GPU/VRAM policies,
  - Evaluation/reward metrics or validation strategy.
- You discover:
  - Stable workarounds for CUDA/FlashAttention/LaTeX quirks on this host,
  - Reliable debugging workflows for LaTeXify.

Store short, titled entries like:

- `LaTeXify: release pipeline contracts v3`
- `LaTeXify: GPU OCR sequencing policy (single 5090)`
- `LaTeXify: reward schema aesthetics tuning`

Each entry should mention:

- What changed,
- Why it changed,
- How it was validated (tests, smoke run on `release/samples/sample.pdf`, metrics). :contentReference[oaicite:8]{index=8}  

If both local `memory` and remote `core_memory` are configured:

- Use `core_memory` for architecture and global rules that should travel across machines.
- Use `memory` for host-specific quirks and experimental notes.

---

## 2. `git` MCP — Source-of-Truth Operations

Use the `git` MCP to stay aligned with the repo’s state.

**Before editing:**

- `git status` to see:
  - Which files are dirty,
  - Whether changes are focused on `release/` and `release/agents/`. :contentReference[oaicite:9]{index=9}  
- `git diff` on relevant paths to understand the latest context.

**While editing:**

- Use `rg` via shell commands to locate symbols and usages instead of broad `find`/`grep`. :contentReference[oaicite:10]{index=10}  

**After editing:**

- Use `git diff` to confirm:
  - Modules stay roughly in the 200–300 line range where possible, :contentReference[oaicite:11]{index=11}  
  - Only intended areas are touched,
  - No generated artifacts or cache files are modified.

Summarize your changes in natural language for commits, PRs, and memory entries.

---

## 3. `sequential_thinking` — Structured Planning

Invoke `sequential_thinking` for any multi-step LaTeXify change.

**Typical triggers:**

- Modifying `release/pipeline/ingestion.py` or other core pipeline modules.
- Changing how agents in `release/agents/` orchestrate the LangGraph-style workflow. :contentReference[oaicite:12]{index=12}  
- Adjusting reward / metrics computation or validation paths.

**Plan contents:**

- Steps with:
  - Goal,
  - Files/modules involved,
  - MCP tools to use (`git`, `memory`, `context7`, `web_search`),
  - Verification method (pytest, smoke run, metrics inspection).

Treat the plan as a working checklist; refine with new calls if reality diverges.

---

## 4. Docs & External Knowledge — `context7`, `web_search`, ResearchAgent

LaTeXify has a RAG/agent stack that already uses external hints via a DuckDuckGo-backed ResearchAgent in `release/agents/`. :contentReference[oaicite:13]{index=13}  

### 4.1 Context7 (PRIMARY for docs & APIs)

**Always use Context7 MCP when you need:**

- Library/API documentation (Python, PyTorch, transformers, LangGraph, LaTeX toolchain, etc.).
- Code generation that depends on accurate library usage (function signatures, kwargs, return types).
- Setup/configuration steps (install commands, environment variables, service configuration).

Protocol:

1. When you need docs or config steps, **call Context7 first**.
2. Ask it to resolve:
   - The correct library ID,
   - The specific docs page(s) relevant to the operation you’re implementing.
3. Only fall back to generic `web_search` if:
   - Context7 doesn’t have the docs, or
   - You need broader discussion / community answers.

### 4.2 Web Search MCP (DuckDuckGo)

Use `web_search` primarily for:

- Debugging runtime/compile errors not covered by internal docs:
  - CUDA, FlashAttention, PyTorch, latexmk/tectonic errors, etc. :contentReference[oaicite:14]{index=14}  
- Finding research papers or repos relevant to:
  - Layout segmentation,
  - OCR and vision-language models,
  - RAG indexing and LaTeX generation.

### 4.3 ResearchAgent (Experimental Agent Stack)

When operating inside the Experimental Agent Stack (Creative → CompileAndRepair → Evaluator → ResearchAgent): :contentReference[oaicite:15]{index=15}  

- Prefer using the built-in `ResearchAgent` for snippet-level hints instead of calling `web_search` directly, so the graph remains consistent.
- Use ResearchAgent after multiple failed snippet attempts to retrieve small supporting examples or explanations.

---

## 5. OpenAI Cookbook — Guided Prompts & Patterns

For advanced prompt design, agent design, or API usage patterns, you should **regularly consult the OpenAI Cookbook**: :contentReference[oaicite:16]{index=16}  

- Primary resources:
  - The GitHub repo `openai/openai-cookbook` (code and notebooks).
  - The website `cookbook.openai.com` (organized examples and guides).

**When to consult the Cookbook:**

- Designing or refining prompts for:
  - Multi-step tool use,
  - RAG flows,
  - Evaluation/reflection loops,
  - Multi-agent workflows similar to LaTeXify’s pipeline.
- Implementing or improving:
  - Retrieval-augmented generation patterns,
  - Function calling / actions,
  - Streaming, batching, or error handling strategies.

**How to consult it:**

- Use `context7` or `web_search` with queries like:
  - `"site:github.com/openai/openai-cookbook RAG"`,
  - `"site:cookbook.openai.com tool use LangGraph"`,
  - `"openai cookbook LaTeX, typesetting, or document pipelines"`.

Adapt patterns from Cookbook examples to the LaTeXify architecture and constraints documented in `release/README.md` and this file. :contentReference[oaicite:17]{index=17}  

---

## 6. `playwright` and `chrome_devtools` — Browser-Level Checks

Use these only when a UI or HTTP surface exists to test.

**Playwright:**

- Script UI flows:
  - Upload a sample PDF,
  - Trigger LaTeXification,
  - Confirm that `main.tex` or output artifacts appear and look sane.

**Chrome DevTools:**

- Inspect performance:
  - Network calls,
  - CPU usage,
  - Console errors.

Don’t use them when the task is purely backend or CLI.

---

## 7. `github` MCP — Remote Issues & PRs

Use `github` to keep local work aligned with remote collaboration:

- Read:
  - Issues labeled for OCR, RAG, agents, metrics, etc.
  - Open PRs touching the same modules you’re changing.
- Write (only when asked):
  - New issues summarizing problems, decisions, and validation.
  - PR descriptions with clear summaries and links to metrics/tests.

---

## 8. Docs vs Memory — What Goes Where

Whenever you modify:

- CLI flags or pipeline entry points,
- Artifact schemas (`graph.json`, `metrics.json`, `quality_gate.json`, etc.),
- Agent contracts or reward schemas,

You must:

1. Update repository docs:
   - `release/README.md` and `release/AGENTS.md` for behavior and agent wiring. :contentReference[oaicite:18]{index=18}  
2. Record a concise memory entry:
   - Name it clearly,
   - Capture reason + impact + validation,
   - So future sessions can recover context via `memory_search`.

If you skip the memory protocol or Context7/Cookbook guidance, you are **not** following the project requirements.
