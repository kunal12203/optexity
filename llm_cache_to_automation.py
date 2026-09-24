"""LLM-based converter: action cache → Optexity automation JSON.

Uses the LLM (via litellm/bedrock) with the Pydantic schema injected as context
so the model understands exactly what to produce.  Validates output with
Automation.model_validate(); on ValidationError it feeds the error back to the
LLM for up to MAX_REPAIR rounds.  Falls back to rule-based cache_to_automation
if all repairs fail.

Usage:
    from llm_cache_to_automation import llm_convert

    automation = llm_convert(
        cache_path="action_cache.json",
        input_parameters={"destination_city": ["Gurgaon"], ...},
        start_url="https://www.booking.com",
    )
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_REPAIR = 3  # self-repair iterations on ValidationError

# ---------------------------------------------------------------------------
# Schema extraction — no hardcoding, sourced directly from Pydantic models
# ---------------------------------------------------------------------------

def _get_focused_schema() -> str:
    """Return a compact JSON schema covering only the action types the LLM needs.

    We omit the full Automation schema (60 defs, very noisy) and instead give the
    LLM a curated subset: the wrapper shape + the five interaction types it will use.
    """
    from optexity.schema.actions.interaction_action import (
        ClickElementAction,
        GoToUrlAction,
        InputTextAction,
        ScrollAction,
        SelectOptionAction,
    )

    wrapper = {
        "automation": {
            "url": "string — start URL",
            "parameters": {
                "input_parameters": "object — same as provided",
                "generated_parameters": {},
            },
            "nodes": "array of action_node objects (see below)",
        },
        "action_node": {
            "type": "action_node",
            "interaction_action": {
                "NOTE": "exactly ONE of the fields below must be set; others omitted",
                "click_element": "ClickElementAction schema",
                "input_text": "InputTextAction schema",
                "select_option": "SelectOptionAction schema",
                "scroll": "ScrollAction schema",
                "go_to_url": "GoToUrlAction schema",
            },
            "expect_new_tab": "bool — set true when click opens a new tab (target=_blank)",
        },
        "ClickElementAction": _trim_schema(ClickElementAction.model_json_schema()),
        "InputTextAction": _trim_schema(InputTextAction.model_json_schema()),
        "SelectOptionAction": _trim_schema(SelectOptionAction.model_json_schema()),
        "ScrollAction": _trim_schema(ScrollAction.model_json_schema()),
        "GoToUrlAction": _trim_schema(GoToUrlAction.model_json_schema()),
    }
    return json.dumps(wrapper, indent=2)


def _trim_schema(schema: dict) -> dict:
    """Keep only 'properties' and 'required' — drop noisy allOf/anyOf/$defs."""
    return {k: v for k, v in schema.items() if k in ("properties", "required", "title")}


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert at converting browser-automation action caches into Optexity \
automation JSON.  You produce valid JSON that matches the provided schema exactly.  \
Never add fields not in the schema.  Return ONLY the JSON object, no prose."""


def _build_prompt(
    cache: dict,
    input_parameters: dict,
    start_url: str,
    schema_str: str,
    prior_error: str | None = None,
    failed_history: str | None = None,
) -> str:
    deterministic = cache.get("deterministic_actions_list", [])

    # Compact element summary for each action
    action_lines = []
    for i, act in enumerate(deterministic):
        el = act.get("element") or {}
        attrs = el.get("attributes", {})
        summary = {
            "i": i,
            "type": act.get("action_type"),
            "params": act.get("action_params", {}),
            "tag": el.get("tag_name", ""),
            "ax_name": el.get("ax_name", ""),
            "id": attrs.get("id", ""),
            "data-test": attrs.get("data-test", ""),
            "data-testid": attrs.get("data-testid", ""),
            "name": attrs.get("name", ""),
            "placeholder": attrs.get("placeholder", ""),
            "role": attrs.get("role", ""),
            "aria-label": attrs.get("aria-label", ""),
            "target": attrs.get("target", ""),
        }
        # Strip empty fields
        summary = {k: v for k, v in summary.items() if v}
        action_lines.append(json.dumps(summary))

    actions_str = "\n".join(action_lines)

    repair_block = ""
    if prior_error:
        repair_block = f"""
The previous attempt produced invalid JSON.  Validation error:
{prior_error}

Fix the error and return the corrected JSON.
"""

    history_block = ""
    if failed_history:
        history_block = f"""
## Failed Node History (locators that were tried and FAILED in previous rounds — do NOT reuse these)
{failed_history}

For each listed node, generate a DIFFERENT locator strategy than what was tried.
"""

    return f"""{repair_block}{history_block}
Convert the following browser action cache into a valid Optexity automation JSON.

## Schema
{schema_str}

## Input Parameters (use {{key[index]}} syntax for variable substitution)
{json.dumps(input_parameters, indent=2)}

## Start URL
{start_url}

## Deterministic Actions (one per line, JSON)
{actions_str}

## Rules
0. Set `max_tries: 2` on every node. Nodes are unproven — fail fast so the
   prompt/agentic fallback layers take over quickly instead of retrying 10×.
   Exception: if a node has `skip_command: true`, max_tries does not matter.
1. Map each action to ONE node with the matching interaction_action type.
2. Use stable Playwright locators in `command`:
   - Prefer: data-testid > id > name > placeholder > role+ax_name > aria-label > xpath
   - For data-testid selectors that match multiple elements append .nth(0)
   - Format: `locator("#id")`, `get_by_role("button", name="Search")`, etc.
3. Replace literal param values with {{key[index]}} refs in both `command` and `input_text`.
4. Set `prompt_instructions` to a short human-readable description of the action.
5. If a click action is followed by a text input on the same element, keep both nodes.
6. Set `expect_new_tab: true` when target="_blank".
7. Set `optional: true` on popup/overlay dismiss nodes.
8. For scroll actions use {{"down": true}} or {{"down": false}}.
9. The `url` field must be the start URL.
10. For `send_keys` actions: map to `input_text` with `command: "locator('body')"`,
    `fill_or_type: "key_press"`, and `input_text` set to the EXACT key string from
    `action_params.keys` (e.g. `"Escape"`, `"Enter"`, `"Tab"`). Never leave `input_text`
    empty for a send_keys action.
11. For `navigate` actions: SKIP — do not create a node for navigation/redirect
    actions (they are side-effects of other actions, not independent steps).
    Exception: the very first navigate to the start URL is already handled by `url`.
12. For `evaluate` actions: SKIP — do not create nodes for JS evaluation actions.
13. For date-picker / calendar cell clicks where the element `ax_name` contains a
    specific date string (e.g. "Fri Sep 25 2026"): set `skip_command: true` (so the
    command layer is skipped) and write `prompt_instructions` describing the action
    using parameter refs, e.g. "Click the check-in date {{checkin_date[0]}} on the calendar".
    Do NOT hardcode the date string in `command`.

Return ONLY a valid JSON object.
"""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(prompt: str) -> str:
    """Call the bedrock model via litellm and return the raw text response."""
    import litellm

    response = litellm.completion(
        model="bedrock/us.anthropic.claude-sonnet-4-6",
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=4096,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | None:
    """Extract the first JSON object from LLM output (handles markdown fences)."""
    # Strip markdown fence
    fenced = re.search(r"```(?:json)?\s*(\{.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)

    # Find outermost { ... }
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
    return None


# ---------------------------------------------------------------------------
# Validate + self-repair
# ---------------------------------------------------------------------------

def _validate(data: dict) -> tuple[object | None, str | None]:
    """Validate against Automation schema.  Returns (instance, None) or (None, error).

    Also handles the common LLM mistake of nesting nodes under an 'automation' key
    instead of placing them at the top level.
    """
    from optexity.schema.automation import Automation

    # Unwrap {"automation": {"nodes": [...]}} if the LLM added an extra wrapper key
    if "nodes" not in data and "automation" in data and isinstance(data["automation"], dict):
        nested = data["automation"]
        if "nodes" in nested:
            data = {k: v for k, v in data.items() if k != "automation"}
            data.update(nested)

    try:
        instance = Automation.model_validate(data)
        # Guard against silent empty-node success — nodes must be present
        if len(instance.nodes) == 0:
            return None, "Automation has 0 nodes — nodes list must not be empty."
        return instance, None
    except Exception as e:
        return None, str(e)[:800]


# ---------------------------------------------------------------------------
# Outcome-driven improver — for round > 0
# ---------------------------------------------------------------------------

_IMPROVE_SYSTEM = """\
You are an expert at making browser-automation scripts more deterministic.
You receive a current Optexity automation JSON plus per-node evidence of what
happened during the last run.  Your job is to improve nodes that are NOT yet
command_success or deterministic, using the evidence provided.
Return ONLY valid JSON — no prose, no markdown fences."""


def _build_improve_prompt(
    current_auto: dict,
    node_evidence: list[dict],
    schema_str: str,
    prior_error: str | None = None,
) -> str:
    """Build the outcome-driven improvement prompt.

    node_evidence is a list of dicts, one per non-deterministic node:
      {
        "node": int,
        "outcome": "prompt_fallback" | "agentic" | "failed",
        "error": str | None,        # command-layer error message
        "winning_locator": str | None,  # LLM locator that worked (prompt_fallback)
        "winning_actions": list[dict],  # actions the agentic actually took (agentic)
      }
    """
    evidence_lines = []
    for e in node_evidence:
        parts = [f"node {e['node']} outcome={e['outcome']}"]
        if e.get("error"):
            parts.append(f"command_error: {e['error'][:150]}")
        if e.get("winning_locator"):
            parts.append(f"winning_locator: {e['winning_locator']}")
        if e.get("winning_actions"):
            acts = e["winning_actions"][:3]  # show first 3 actions
            acts_compact = [
                {
                    "type": a.get("action_type"),
                    "el_id": (a.get("element") or {}).get("attributes", {}).get("id", ""),
                    "el_testid": (a.get("element") or {}).get("attributes", {}).get("data-testid", ""),
                    "el_placeholder": (a.get("element") or {}).get("attributes", {}).get("placeholder", ""),
                    "el_ax": (a.get("element") or {}).get("ax_name", "")[:40],
                    "text": a.get("action_params", {}).get("text", ""),
                    "keys": a.get("action_params", {}).get("keys", ""),
                }
                for a in acts
            ]
            parts.append(f"winning_actions: {json.dumps(acts_compact)}")
        evidence_lines.append("  " + " | ".join(parts))

    evidence_str = "\n".join(evidence_lines) if evidence_lines else "  (none)"

    repair_block = ""
    if prior_error:
        repair_block = f"\nPrevious attempt was invalid. Validation error:\n{prior_error}\nFix it.\n"

    return f"""{repair_block}
Improve the following Optexity automation JSON so that non-deterministic nodes
become deterministic.  Use the evidence below to guide each fix.

## Schema
{schema_str}

## Current automation
{json.dumps(current_auto, indent=2)}

## Evidence per non-deterministic node
{evidence_str}

## General rules
- Set `max_tries: 2` on all nodes you modify or add. Fail fast — let prompt/agentic
  layers take over quickly rather than retrying 10× on a wrong locator.
- Nodes that already have outcome=command_success or deterministic: do NOT change them,
  including their max_tries.

## How to use the evidence
- outcome=failed, error contains "covered by another element" OR "intercepts pointer events" OR "Timeout" (element resolved but blocked):
    → Also set force: true on the failing click node.
    → If it's a coverage/overlap issue, also insert an optional Escape key-press node BEFORE the failing node
      (command: locator('body'), fill_or_type: key_press, input_text: Escape, optional: true)
- outcome=failed, error contains "strict mode violation" or "resolved to N elements":
    → Append .nth(0) to the command locator.
- outcome=failed, error contains "not found" or "timeout":
    → Replace command with a more stable locator derived from winning_actions elements
      (prefer id > data-testid > placeholder > aria-label > role+name).
- outcome=prompt_fallback, winning_locator provided:
    → Replace the node's command with that winning_locator exactly.
- outcome=agentic, winning_actions provided:
    → Rebuild the node using the first meaningful action's element attributes
      to form a stable Playwright locator.
    → For send_keys actions in winning_actions: add an input_text key_press node.
    → For date-picker clicks (ax_name looks like a calendar date): set skip_command: true,
      write prompt_instructions with parameter refs instead of hardcoded dates.
- For any SEARCH / SUBMIT / final form-submit button: set force: true.

## Critical structural rules
- REPLACE failing nodes in place — do NOT append new nodes for goals already served by existing nodes.
- If two nodes in the current automation serve the same goal (e.g., two city-input nodes), keep only the better one and remove the duplicate.
- NEVER hardcode dates or specific text in locator commands (e.g., has_text="Mon Sep 28 2026"). Any calendar/date-picker click MUST use skip_command: true and parameter refs in prompt_instructions.

Return the COMPLETE improved automation JSON (all nodes, including unchanged ones).
"""


def llm_improve(
    current_auto: dict,
    node_evidence: list[dict],
) -> dict:
    """Improve a current automation using per-node outcome evidence.

    node_evidence: list from build_node_evidence().
    Returns improved automation dict.  Falls back to current_auto on failure.
    """
    if not node_evidence:
        return current_auto  # nothing to improve

    schema_str = _get_focused_schema()
    prior_error: str | None = None

    for attempt in range(MAX_REPAIR + 1):
        if attempt > 0:
            logger.info(f"[LLM improver] repair attempt {attempt}/{MAX_REPAIR}")

        prompt = _build_improve_prompt(current_auto, node_evidence, schema_str, prior_error)
        try:
            raw = _call_llm(prompt)
        except Exception as e:
            logger.error(f"[LLM improver] LLM call failed: {e}")
            break

        data = _extract_json(raw)
        if data is None:
            prior_error = "Response did not contain a valid JSON object."
            continue

        instance, error = _validate(data)
        if instance is not None:
            logger.info(f"[LLM improver] validated on attempt {attempt}")
            return instance.model_dump(mode="json", exclude_none=True)

        prior_error = error
        logger.warning(f"[LLM improver] attempt {attempt}: {error}")

    logger.warning("[LLM improver] all attempts failed — returning current automation unchanged.")
    return current_auto


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def llm_convert(
    cache_path: str | Path,
    input_parameters: dict | None = None,
    start_url: str | None = None,
    failed_history: str | None = None,
) -> dict:
    """Build a deterministic Optexity automation from an action cache using LLM.

    failed_history: compact multi-line string of previously-failed (node, locator, error)
    entries so the LLM avoids repeating the same broken locators.

    Falls back to rule-based cache_to_automation on failure.
    Returns a plain dict (not a Pydantic model).
    """
    cache_path = Path(cache_path)
    with open(cache_path) as f:
        cache = json.load(f)

    url = start_url or cache.get("start_url", "")
    params = input_parameters or {}
    schema_str = _get_focused_schema()

    prior_error: str | None = None
    for attempt in range(MAX_REPAIR + 1):
        if attempt > 0:
            logger.info(f"[LLM builder] repair attempt {attempt}/{MAX_REPAIR}")

        prompt = _build_prompt(cache, params, url, schema_str, prior_error, failed_history)
        try:
            raw = _call_llm(prompt)
        except Exception as e:
            logger.error(f"[LLM builder] LLM call failed: {e}")
            break

        data = _extract_json(raw)
        if data is None:
            prior_error = "Response did not contain a valid JSON object."
            logger.warning(f"[LLM builder] attempt {attempt}: no JSON found")
            continue

        instance, error = _validate(data)
        if instance is not None:
            logger.info(f"[LLM builder] validated on attempt {attempt}")
            # Return the instance serialized back to dict so any unwrapping done
            # in _validate (e.g. nested "automation" key) is reflected in output.
            return instance.model_dump(mode="json", exclude_none=True)

        prior_error = error
        logger.warning(f"[LLM builder] attempt {attempt}: validation failed — {error}")

    # All attempts exhausted — fall back to rule-based
    logger.warning("[LLM builder] falling back to rule-based converter")
    from cache_to_automation import convert_cache_to_automation

    return convert_cache_to_automation(cache_path, params, url)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="LLM-based cache → automation converter")
    parser.add_argument("--cache", required=True, help="Path to action_cache.json")
    parser.add_argument("--params", default=None, help="JSON file with input_parameters")
    parser.add_argument("--output", default="automation_llm.json", help="Output path")
    parser.add_argument("--url", default=None, help="Override start URL")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    input_parameters: dict = {}
    if args.params:
        with open(args.params) as f:
            p = json.load(f)
        input_parameters = (
            p.get("input_parameters")
            or p.get("parameters", {}).get("input_parameters", {})
            or p
        )

    automation = llm_convert(args.cache, input_parameters, args.url)

    with open(args.output, "w") as f:
        json.dump(automation, f, indent=2)

    n = len(automation.get("nodes", []))
    print(f"Generated {n} nodes → {args.output}")

    from optexity.schema.automation import Automation

    try:
        Automation.model_validate(automation)
        print("Schema validation: PASSED")
    except Exception as e:
        print(f"Schema validation: FAILED — {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
