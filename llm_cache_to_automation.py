"""LLM-based converter: action cache → Optexity automation JSON.

Uses the LLM (via litellm) with the Pydantic schema injected as context
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
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_REPAIR = 3


# ---------------------------------------------------------------------------
# Schema (cached — Pydantic schema doesn't change at runtime)
# ---------------------------------------------------------------------------

def _trim_schema(schema: dict) -> dict:
    return {k: v for k, v in schema.items() if k in ("properties", "required", "title")}


@lru_cache(maxsize=1)
def _get_focused_schema() -> str:
    from optexity.schema.actions.interaction_action import (
        ClickElementAction,
        GoToUrlAction,
        InputTextAction,
        KeyPressAction,
        ScrollAction,
        SelectOptionAction,
        SwitchTabAction,
    )

    wrapper = {
        "automation": {
            "url": "string — start URL",
            "parameters": {"input_parameters": "object — same as provided", "generated_parameters": {}},
            "nodes": "array of action_node objects (see below)",
        },
        "action_node": {
            "type": "action_node",
            "interaction_action": {
                "NOTE": "exactly ONE of the fields below must be set; others omitted",
                "click_element": "ClickElementAction",
                "input_text": "InputTextAction",
                "select_option": "SelectOptionAction",
                "scroll": "ScrollAction",
                "go_to_url": "GoToUrlAction",
                "key_press": "KeyPressAction — for send_keys/keyboard actions",
                "switch_tab": "SwitchTabAction — for tab switch actions",
            },
            "expect_new_tab": "bool — true when click opens a new tab",
        },
        "ClickElementAction": _trim_schema(ClickElementAction.model_json_schema()),
        "InputTextAction": _trim_schema(InputTextAction.model_json_schema()),
        "SelectOptionAction": _trim_schema(SelectOptionAction.model_json_schema()),
        "ScrollAction": _trim_schema(ScrollAction.model_json_schema()),
        "GoToUrlAction": _trim_schema(GoToUrlAction.model_json_schema()),
        "KeyPressAction": _trim_schema(KeyPressAction.model_json_schema()),
        "SwitchTabAction": _trim_schema(SwitchTabAction.model_json_schema()),
    }
    return json.dumps(wrapper, indent=2)


# ---------------------------------------------------------------------------
# LLM call + JSON extraction + validation + repair loop
# ---------------------------------------------------------------------------

def _resolve_model() -> str:
    import os
    env_model = os.environ.get("LLM_MODEL")
    if env_model:
        return env_model
    try:
        from optexity.utils.llm_settings import llm_settings
        if llm_settings.LLM_MODEL:
            return llm_settings.LLM_MODEL
    except Exception:
        pass
    return "bedrock/us.anthropic.claude-sonnet-4-6"


def _call_llm(system: str, prompt: str) -> str:
    import litellm

    response = litellm.completion(
        model=_resolve_model(),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=4096,
    )
    return response.choices[0].message.content or ""


def _extract_json(text: str) -> dict | None:
    # Fast path: raw JSON response
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Strip markdown fence
    fenced = re.search(r"```(?:json)?\s*(\{.*?)\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # Brace-matching fallback
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
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    pass
    return None


def _semantic_check(data: dict, input_parameters: dict | None = None) -> str | None:
    """Check for issues that pass schema validation but break at runtime."""
    errors = []
    params = input_parameters or {}
    param_values = set()
    for vals in params.values():
        if isinstance(vals, list):
            for v in vals:
                if v and len(str(v)) >= 2:
                    param_values.add(str(v).lower())

    for i, node in enumerate(data.get("nodes", [])):
        ia = node.get("interaction_action", {})
        for action_type in ("input_text", "click_element", "select_option", "key_press"):
            action = ia.get(action_type)
            if not isinstance(action, dict):
                continue
            cmd = action.get("command", "")
            if cmd.startswith("page."):
                errors.append(f"Node {i}: command starts with 'page.' — remove it, runtime prepends page. automatically.")
            text = action.get("input_text", "")
            if text and "{" not in text and param_values:
                text_lower = text.lower()
                for pv in param_values:
                    if pv in text_lower or text_lower in pv:
                        errors.append(f"Node {i}: input_text '{text}' looks like a param value but is hardcoded — use {{key[index]}} ref.")
                        break

    if errors:
        return " | ".join(errors[:5])
    return None


def _validate(data: dict, input_parameters: dict | None = None) -> tuple[object | None, str | None]:
    from optexity.schema.automation import Automation

    # Unwrap {"automation": {...}} if the LLM added an extra wrapper
    if "nodes" not in data and "automation" in data and isinstance(data["automation"], dict):
        nested = data["automation"]
        if "nodes" in nested:
            data = {k: v for k, v in data.items() if k != "automation"}
            data.update(nested)

    try:
        instance = Automation.model_validate(data)
        if len(instance.nodes) == 0:
            return None, "Automation has 0 nodes — nodes list must not be empty."
    except Exception as e:
        return None, str(e)[:800]

    semantic_err = _semantic_check(data, input_parameters)
    if semantic_err:
        return None, f"Semantic issues: {semantic_err}"

    return instance, None


def _llm_with_repair(
    system: str,
    build_prompt,
    label: str,
    fallback=None,
    input_parameters: dict | None = None,
) -> dict | None:
    """Shared repair loop: call LLM → extract JSON → validate (schema + semantic) → retry on error."""
    prior_error: str | None = None

    for attempt in range(MAX_REPAIR + 1):
        if attempt > 0:
            logger.info(f"[{label}] repair attempt {attempt}/{MAX_REPAIR}")

        prompt = build_prompt(prior_error)
        try:
            raw = _call_llm(system, prompt)
        except Exception as e:
            logger.error(f"[{label}] LLM call failed: {e}")
            break

        data = _extract_json(raw)
        if data is None:
            prior_error = "Response did not contain a valid JSON object."
            logger.warning(f"[{label}] attempt {attempt}: no JSON found")
            continue

        instance, error = _validate(data, input_parameters)
        if instance is not None:
            logger.info(f"[{label}] validated on attempt {attempt}")
            return instance.model_dump(mode="json", exclude_none=True)

        prior_error = error
        logger.warning(f"[{label}] attempt {attempt}: {error}")

    if fallback:
        logger.warning(f"[{label}] all attempts failed — using fallback")
        return fallback()

    return None


# ---------------------------------------------------------------------------
# Param annotation
# ---------------------------------------------------------------------------

def _build_reverse_param_map(input_parameters: dict) -> dict[str, str]:
    """Build lowercase-value → {key[index]} map for pre-annotating actions."""
    reverse: dict[str, str] = {}
    pairs = []
    for key, values in input_parameters.items():
        if not isinstance(values, list):
            values = [values]
        for idx, val in enumerate(values):
            val_str = str(val).strip()
            if val_str:
                pairs.append((val_str, f"{{{key}[{idx}]}}"))
    # Longer values first so "John Doe" matches full_name before "John" matches first_name
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    for val_str, ref in pairs:
        if val_str.lower() not in reverse:
            reverse[val_str.lower()] = ref
    return reverse


# ---------------------------------------------------------------------------
# Convert prompt (round 0: cache → automation)
# ---------------------------------------------------------------------------

_CONVERT_SYSTEM = """\
You convert browser-automation action caches into Optexity automation JSON.
Produce valid JSON matching the schema exactly. No extra fields. Return ONLY JSON, no prose."""


def _build_convert_prompt(
    cache: dict,
    input_parameters: dict,
    start_url: str,
    schema_str: str,
    failed_history: str | None = None,
) -> callable:
    deterministic = cache.get("deterministic_actions_list", [])
    reverse_params = _build_reverse_param_map(input_parameters)

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
        typed_text = str(act.get("action_params", {}).get("text", "")).strip()
        if typed_text and typed_text.lower() in reverse_params:
            summary["param_ref"] = reverse_params[typed_text.lower()]
        summary = {k: v for k, v in summary.items() if v}
        action_lines.append(json.dumps(summary, separators=(",", ":")))

    actions_str = "\n".join(action_lines)
    params_str = json.dumps(input_parameters, separators=(",", ":"))

    history_block = ""
    if failed_history:
        history_block = f"\n## Failed locator history (do NOT reuse these)\n{failed_history}\n"

    def prompt_fn(prior_error: str | None = None) -> str:
        repair = ""
        if prior_error:
            repair = f"Previous attempt invalid: {prior_error}\nFix and return corrected JSON.\n\n"

        return f"""{repair}{history_block}Convert this action cache into Optexity automation JSON.

## Schema
{schema_str}

## Input Parameters (use {{key[index]}} for variable substitution)
{params_str}

## Start URL
{start_url}

## Actions
{actions_str}

## Rules

### Structure
1. One action → one node. Set `max_tries: 2` on every node.
2. Set `prompt_instructions` to a short description of the action.
3. Set `expect_new_tab: true` when target="_blank"; `optional: true` on popup dismissals.

### Locators
4. `command` must be a Playwright locator WITHOUT `page.` prefix (the runtime prepends it).
   Correct: `locator('[name="field"]')`, `get_by_role("button", name="X")`.
   WRONG: `page.locator(...)`.
   Priority: data-testid > id > name > placeholder > role+ax_name > aria-label > xpath. Append `.nth(0)` when multiple matches.

### Variables
5. NEVER hardcode param values. Use {{key[index]}} refs everywhere. When `param_ref` is present, use it exactly.
   If the typed text doesn't match any param but the field name suggests a param (e.g. field "04lastname" → last_name), use the param ref anyway.
6. Date-pickers/calendar cells: set `skip_command: true`, use param refs in `prompt_instructions`.

### Type mapping
7. input → input_text; click → click_element; select_dropdown → select_option; scroll → scroll (down: true/false); send_keys → key_press (type = key string); switch → switch_tab (tab_index from page_id, default 0).

### Skip (no node)
8. navigate, evaluate, extract, search, write_file, done, wait — not replayable.

Return ONLY valid JSON."""

    return prompt_fn


# ---------------------------------------------------------------------------
# Improve prompt (round > 0: evidence-driven fixes)
# ---------------------------------------------------------------------------

_IMPROVE_SYSTEM = """\
You improve browser-automation scripts using per-node evidence.
Fix non-deterministic nodes. Return ONLY valid JSON, no prose."""

# Evidence → fix mapping: generalized from Playwright error patterns
_EVIDENCE_RULES = """## Evidence → fix mapping
- error mentions element blocked/covered/intercepted/overlapped → set `force: true`; optionally insert Escape key_press before node.
- error mentions strict mode / resolved to N elements → append `.nth(0)` to locator.
- error mentions not found / timeout with no element match → replace locator from winning_actions or winning_locator attributes.
- outcome=prompt_fallback + winning_locator → use that locator as the command.
- outcome=agentic + winning_actions → rebuild locator from first action's element attributes.
- date-picker actions → skip_command: true, param refs in prompt_instructions.
- submit/search buttons → set force: true."""


def _build_improve_prompt_fn(
    current_auto: dict,
    node_evidence: list[dict],
    schema_str: str,
) -> callable:
    evidence_lines = []
    for e in node_evidence:
        parts = [f"node {e['node']} outcome={e['outcome']}"]
        if e.get("error"):
            parts.append(f"error: {e['error'][:150]}")
        if e.get("winning_locator"):
            parts.append(f"winning_locator: {e['winning_locator']}")
        if e.get("winning_actions"):
            acts_compact = [
                {k: v for k, v in {
                    "type": a.get("action_type"),
                    "id": (a.get("element") or {}).get("attributes", {}).get("id", ""),
                    "testid": (a.get("element") or {}).get("attributes", {}).get("data-testid", ""),
                    "ax": (a.get("element") or {}).get("ax_name", "")[:40],
                    "text": a.get("action_params", {}).get("text", ""),
                }.items() if v}
                for a in e["winning_actions"][:3]
            ]
            parts.append(f"actions: {json.dumps(acts_compact, separators=(',', ':'))}")
        evidence_lines.append("  " + " | ".join(parts))

    evidence_str = "\n".join(evidence_lines) if evidence_lines else "  (none)"
    auto_str = json.dumps(current_auto, separators=(",", ":"))

    def prompt_fn(prior_error: str | None = None) -> str:
        repair = ""
        if prior_error:
            repair = f"Previous attempt invalid: {prior_error}\nFix it.\n\n"

        return f"""{repair}Improve this automation using the evidence below.

## Schema
{schema_str}

## Current automation
{auto_str}

## Evidence
{evidence_str}

{_EVIDENCE_RULES}

## Structural rules
- Set `max_tries: 2` on modified nodes. Don't touch nodes with outcome=command_success or deterministic.
- REPLACE failing nodes in place — no duplicates for the same goal.
- NEVER hardcode dates in locators. Use skip_command: true + param refs.

Return the COMPLETE automation JSON (all nodes, including unchanged ones)."""

    return prompt_fn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def llm_convert(
    cache_path: str | Path,
    input_parameters: dict | None = None,
    start_url: str | None = None,
    failed_history: str | None = None,
) -> dict:
    cache_path = Path(cache_path)
    with open(cache_path) as f:
        cache = json.load(f)

    url = start_url or cache.get("start_url", "")
    params = input_parameters or {}
    schema_str = _get_focused_schema()

    prompt_fn = _build_convert_prompt(cache, params, url, schema_str, failed_history)

    def fallback():
        from cache_to_automation import convert_cache_to_automation
        return convert_cache_to_automation(cache_path, params, url)

    result = _llm_with_repair(_CONVERT_SYSTEM, prompt_fn, "LLM builder", fallback, input_parameters=params)
    return result or fallback()


def llm_improve(
    current_auto: dict,
    node_evidence: list[dict],
) -> dict:
    if not node_evidence:
        return current_auto

    schema_str = _get_focused_schema()
    prompt_fn = _build_improve_prompt_fn(current_auto, node_evidence, schema_str)

    result = _llm_with_repair(
        _IMPROVE_SYSTEM, prompt_fn, "LLM improver",
        fallback=lambda: current_auto,
    )
    return result or current_auto


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
