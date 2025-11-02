# Prompt Test Scenarios

These examples illustrate how each agent should respond to the v1.0 prompts. Use them to validate structure, tone, and KB propagation reminders.

## Phase 2 Planner

### Sample Prompt 1
```
Inputs: plan context JSON (v0.9 legacy), OCR blocks (2 issues), retrieval hits (repo + KB), compile logs (missing bibliography file).
Goal: produce updated remediation roadmap.
```

### Sample Response 1
```json
{
  "status": "ok",
  "plan_version": "phase2-planner/v1.0",
  "plan_summary": "Coordinate bibliography fixes and regenerate figures.",
  "milestones": [
    {
      "id": "M1",
      "objective": "Restore bibliography pipeline",
      "inputs": ["plan-context:legacy-v0.9", "compile-log:2024-06-01"],
      "actions": ["Confirm missing .bib file location", "Assign retrieval agent to surface previous citation fixes"],
      "owner": "retrieval-agent",
      "kb_propagation": "Once resolved, add the citation repair steps to KB::latex/bibtex/v1 for non-OpenAI reuse"
    }
  ],
  "risks": [
    {
      "id": "R1",
      "description": "Missing .bib file persists after fetch",
      "mitigation": "Escalate to self-heal loop to regenerate inputs"
    }
  ],
  "handoff_notes": "Retrieval should reuse prior bibliography playbooks; synthesis must confirm KB propagation steps.",
  "kb_actions": [
    {
      "type": "record_fix",
      "trigger": "Bibliography build succeeds twice",
      "destination": "KB::latex/bibtex/v1"
    }
  ]
}
```

### Sample Prompt 2
```
Inputs: missing plan context JSON.
Goal: attempt to plan.
```

### Sample Response 2
```json
{
  "status": "cannot_comply",
  "plan_version": "phase2-planner/v1.0",
  "reason": "Required plan context JSON missing.",
  "next_step": "Escalate back to the compile/self-heal loop for regeneration of inputs."
}
```

## Retrieval Agent

### Sample Prompt 1
```
Inputs: planner milestone M1 about bibliography, OCR blocks highlighting citation placeholders, retrieval hits (outdated search), compile log showing BibTeX error.
Goal: produce targeted searches.
```

### Sample Response 1
```json
{
  "status": "ok",
  "prompt_version": "retrieval-agent/v1.0",
  "search_intent": "Locate prior fixes for missing bibliography artifacts and relevant repo configuration files.",
  "queries": [
    {
      "id": "Q1",
      "channel": "kb",
      "query": "bibliography missing .bib file remediation",
      "rationale": "Planner milestone M1 calls for KB-backed citation fixes"
    }
  ],
  "selected_hits": [
    {
      "query_id": "Q1",
      "source_path": "KB::latex/bibtex/v0.8",
      "excerpt": "Verify that references.bib is copied during build stage.",
      "confidence": "medium",
      "kb_propagation": "Update entry to v1 once fix verified for non-OpenAI compatible agents"
    }
  ],
  "gaps": [
    {
      "description": "Need confirmation of repo-level copy script",
      "follow_up": "Run repo search for references.bib in CI pipeline"
    }
  ]
}
```

### Sample Prompt 2
```
Inputs: plan context JSON absent.
Goal: gather retrieval guidance.
```

### Sample Response 2
```json
{
  "status": "cannot_comply",
  "prompt_version": "retrieval-agent/v1.0",
  "reason": "Planner context missing; cannot align retrieval with milestones.",
  "next_step": "Hand control back to the compile/self-heal loop for input regeneration."
}
```

## Synthesis Agent

### Sample Prompt 1
```
Inputs: planner milestone M1, OCR block describing broken citations, retrieval hit KB::latex/bibtex/v0.8, compile log showing missing references.bib.
Goal: craft actionable fix.
```

### Sample Response 1
```json
{
  "status": "ok",
  "prompt_version": "synthesis-agent/v1.0",
  "solution_summary": "Restore references.bib copy step and regenerate bibliography cache.",
  "changes": [
    {
      "id": "C1",
      "target": "scripts/build.sh",
      "action": "edit",
      "instructions": [
        "Add cp references.bib build/refs/references.bib before latexmk",
        "Document the step in README build instructions"
      ],
      "justification": "Compile log shows missing file; retrieval confirms previous fix pattern.",
      "kb_propagation": "Record the updated script in KB::latex/bibtex/v1 with version tags for non-OpenAI agents"
    }
  ],
  "validation": [
    {
      "test": "nix develop --command latexmk -pdf main.tex",
      "purpose": "Ensure bibliography compiles after script change"
    }
  ],
  "handoff": "Self-heal loop should apply script edit, rerun compile, and publish KB entry when stable."
}
```

### Sample Prompt 2
```
Inputs: retrieval hits contradict compile log (file already present).
Goal: propose fix.
```

### Sample Response 2
```json
{
  "status": "cannot_comply",
  "prompt_version": "synthesis-agent/v1.0",
  "reason": "Evidence conflict between retrieval hits and compile log prevents confident synthesis.",
  "next_step": "Return control to the compile/self-heal loop for additional planning or retrieval."
}
```

## JSON Sanity Check
Use Python to validate sample responses:

```bash
python -m json.tool <<<'{"status": "ok", "plan_version": "phase2-planner/v1.0", "milestones": [], "risks": [], "handoff_notes": "", "kb_actions": []}'
```
