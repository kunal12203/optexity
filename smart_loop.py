"""Smart iterative loop: agentic → cache → LLM build → replay → improve → repeat.

Each round:
  1. Run the current automation (or 2-node agentic for round 0).
  2. Read node_outcomes.json to classify each node's execution path.
  3. For prompt_fallback nodes: extract the winning LLM locator from optexity.log
     and promote it to `command` for the next round.
  4. For failed nodes: revert to an agentic_task sub-step.
  5. For command_success / deterministic nodes: lock them (no change).
  6. Use the LLM builder on the latest merged cache to produce the next automation.
  7. Repeat until all nodes are command_success / deterministic, or max_rounds.

Usage:
    python smart_loop.py \\
        --endpoint reserve_hotel_booking_com-074ae7dd \\
        --params booking_params.json \\
        --url https://www.booking.com \\
        --agentic-task "Search for hotels in {destination_city[0]}..." \\
        --max-rounds 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SERVER_URL = "http://localhost:9000"
POLL_INTERVAL = 10   # seconds between status checks
MAX_WAIT = 900       # max seconds to wait for a task

# ---------------------------------------------------------------------------
# Task execution helpers
# ---------------------------------------------------------------------------

def trigger_task(endpoint: str, input_parameters: dict, timeout_min: int = 15) -> str | None:
    """POST to /inference and return the task_id."""
    payload = {
        "endpoint_name": endpoint,
        "input_parameters": input_parameters,
        "max_timeout_in_minutes": timeout_min,
    }
    try:
        resp = requests.post(f"{SERVER_URL}/inference", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        task_id = data.get("task_id") or data.get("id")
        logger.info(f"Task triggered: {task_id}")
        return task_id
    except Exception as e:
        logger.error(f"Failed to trigger task: {e}")
        return None


def wait_for_task(task_id: str) -> dict | None:
    """Wait for a task to finish by watching the local log file.

    The local server has no /task/<id> endpoint, so we watch:
      1. /tmp/optexity/<task_id>/logs/optexity.log for "completed with status"
      2. /health to confirm task_running flipped back to False
    Returns {"status": "success"|"failed"} or None on timeout.
    """
    log_path = Path(f"/tmp/optexity/{task_id}/logs/optexity.log")
    deadline = time.time() + MAX_WAIT

    while time.time() < deadline:
        # Check log file for final status line
        if log_path.exists():
            try:
                text = log_path.read_text(errors="replace")
                for status in ("success", "failed"):
                    if f"completed with status {status}" in text:
                        logger.info(f"Task {task_id} finished: {status}")
                        return {"status": status, "task_id": task_id}
            except Exception:
                pass
        time.sleep(POLL_INTERVAL)

    logger.error(f"Task {task_id} timed out after {MAX_WAIT}s")
    return None


def find_task_dir(task_id: str, base: str = "/tmp/optexity") -> Path | None:
    p = Path(base) / task_id
    return p if p.exists() else None


# ---------------------------------------------------------------------------
# Outcome reading
# ---------------------------------------------------------------------------

def read_node_outcomes(task_dir: Path) -> list[dict]:
    path = task_dir / "logs" / "node_outcomes.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def outcomes_summary(outcomes: list[dict]) -> dict[str, int]:
    from collections import Counter
    return dict(Counter(o["outcome"] for o in outcomes))


def all_deterministic(outcomes: list[dict]) -> bool:
    return all(
        o["outcome"] in ("command_success", "deterministic", "skipped")
        for o in outcomes
    )


# ---------------------------------------------------------------------------
# Log parsing: extract winning LLM locator + failure errors per node
# ---------------------------------------------------------------------------

_LLM_FALLBACK_RE = re.compile(
    r"LLM fallback locator \[index \d+\]: (page\.[^\s(]+\([^)]*\)(?:\.[^\s(]+\([^)]*\))*)"
)
_NODE_START_RE = re.compile(r"-----Running node new (\d+)-----")
# "ClickElementAction failed after 10 tries: error: <message>"
_ACTION_FAIL_RE = re.compile(
    r"(\w+Action) failed after \d+ tries: (.+?)$"
)


def extract_fallback_locators(log_path: Path) -> dict[int, str]:
    """Parse optexity.log and return {node_index: winning_locator_command}.

    The locator is formatted as `page.get_by_role(...)` — we strip the leading
    `page.` so it can be used directly as an Optexity `command` value.
    Also strips the trailing `.click(...)` / `.fill(...)` method call since
    Optexity appends the method itself.
    """
    if not log_path.exists():
        return {}

    locators: dict[int, str] = {}
    current_node: int | None = None

    for line in log_path.read_text(errors="replace").splitlines():
        node_match = _NODE_START_RE.search(line)
        if node_match:
            current_node = int(node_match.group(1))
            continue

        fallback_match = _LLM_FALLBACK_RE.search(line)
        if fallback_match and current_node is not None:
            full_locator = fallback_match.group(1)
            # Strip leading "page." and trailing method call like ".click(...)" / ".fill(...)"
            cmd = re.sub(r"^page\.", "", full_locator)
            cmd = re.sub(r"\.(click|fill|select_option|check|uncheck)\([^)]*\)$", "", cmd)
            if current_node not in locators:  # keep first (highest-confidence)
                locators[current_node] = cmd

    return locators


def extract_failed_errors(log_path: Path) -> dict[int, str]:
    """Parse optexity.log and return {node_index: short_error} for failed nodes.

    Captures the first failure per node plus any 'intercepts pointer events' detail
    from the Playwright call log on subsequent lines (element coverage errors).
    """
    if not log_path.exists():
        return {}

    errors: dict[int, str] = {}
    current_node: int | None = None
    lines = log_path.read_text(errors="replace").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        node_match = _NODE_START_RE.search(line)
        if node_match:
            current_node = int(node_match.group(1))
            i += 1
            continue

        fail_match = _ACTION_FAIL_RE.search(line)
        if fail_match and current_node is not None and current_node not in errors:
            raw_error = fail_match.group(2).strip()
            short = raw_error.split("\n")[0][:120]
            # Scan ahead into the Playwright call log for coverage detail
            j = i + 1
            while j < len(lines):
                ahead = lines[j]
                if _NODE_START_RE.search(ahead) or "-----Finished node" in ahead:
                    break
                if "intercepts pointer events" in ahead:
                    short += " [element covered: intercepts pointer events]"
                    break
                j += 1
            errors[current_node] = short
        i += 1

    return errors


def build_failed_history(
    round_num: int,
    automation: dict,
    outcomes: list[dict],
    failed_errors: dict[int, str],
    history_path: Path,
) -> list[dict]:
    """Append failed-node records to a persistent loop_history.json.

    Only records nodes whose outcome is 'failed' — prompt_fallback and
    command_success nodes don't need history (they either have a winning
    locator or are already locked).
    Returns the full accumulated history list.
    """
    existing: list[dict] = []
    if history_path.exists():
        try:
            existing = json.loads(history_path.read_text())
        except Exception:
            pass

    outcome_by_node = {o["node"]: o["outcome"] for o in outcomes}
    nodes = automation.get("nodes", [])

    for idx, node in enumerate(nodes):
        if outcome_by_node.get(idx) != "failed":
            continue
        ia = node.get("interaction_action", {})
        tried_cmd = ""
        for atype in ("click_element", "input_text", "select_option"):
            action = ia.get(atype)
            if action:
                tried_cmd = action.get("command", "")
                break
        existing.append({
            "round": round_num,
            "node": idx,
            "tried_command": tried_cmd,
            "error": failed_errors.get(idx, "unknown error"),
        })

    history_path.write_text(json.dumps(existing, indent=2))
    return existing


def format_failed_history(history: list[dict]) -> str:
    """Render failed history as compact lines for the LLM prompt.

    Example output:
      [r1, n5] tried: locator('[data-testid="hotel"]') → strict mode violation: resolved to 15 elements
      [r2, n5] tried: get_by_test_id("title-link").nth(0) → TimeoutError: element not found
    """
    if not history:
        return ""
    lines = [
        f"[r{h['round']}, n{h['node']}] tried: {h['tried_command']} → {h['error']}"
        for h in history
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Automation patching based on outcomes
# ---------------------------------------------------------------------------

def _action_to_command(action: dict) -> str | None:
    """Build a stable Playwright locator string from a cached action's element data.

    Priority: id > data-testid/data-cy > placeholder > aria-label > role+name > xpath.
    Returns None if no usable selector can be derived.
    """
    el = action.get("element") or {}
    attrs = el.get("attributes", {})
    tag = el.get("tag_name", "")
    ax_name = el.get("ax_name", "")

    if attrs.get("id"):
        return f'locator("#{attrs["id"]}")'
    if attrs.get("data-testid"):
        return f'locator(\'[data-testid="{attrs["data-testid"]}"]\').nth(0)'
    if attrs.get("data-cy"):
        return f'locator(\'[data-cy="{attrs["data-cy"]}"]\').nth(0)'
    if attrs.get("placeholder"):
        return f'get_by_placeholder("{attrs["placeholder"]}")'
    if attrs.get("aria-label"):
        return f'get_by_label("{attrs["aria-label"]}")'
    if ax_name and attrs.get("role"):
        return f'get_by_role("{attrs["role"]}", name="{ax_name}")'
    if ax_name and tag in ("button", "a", "input", "select"):
        role = {"button": "button", "a": "link", "input": "textbox", "select": "combobox"}.get(tag, tag)
        return f'get_by_role("{role}", name="{ax_name}")'
    xpath = el.get("xpath", "")
    if xpath:
        return f'locator("xpath={xpath}")'
    return None


def patch_automation(
    automation: dict,
    outcomes: list[dict],
    fallback_locators: dict[int, str],
    agentic_task_text: str,
    input_parameters: dict,
    step_caches: dict[int, list[dict]] | None = None,
) -> dict:
    """Return an updated automation dict based on per-node outcomes.

    Strategy per node:
    - command_success / deterministic / skipped → keep as-is (locked)
    - prompt_fallback → promote winning locator from log to command
    - agentic → if step_caches has actions for this node, promote to deterministic command
    - failed → replace interaction_action with agentic_task fallback
    """
    import copy

    patched = copy.deepcopy(automation)
    nodes = patched.get("nodes", [])
    step_caches = step_caches or {}

    outcome_by_node: dict[int, str] = {o["node"]: o["outcome"] for o in outcomes}

    for idx, node in enumerate(nodes):
        outcome = outcome_by_node.get(idx, "unknown")

        if outcome in ("command_success", "deterministic", "skipped"):
            # Proven — lock max_tries high so the command layer always runs reliably.
            ia = node.get("interaction_action", {})
            if "max_tries" in ia:
                ia["max_tries"] = 10
            continue

        if outcome == "prompt_fallback":
            winning = fallback_locators.get(idx)
            if winning:
                ia = node.get("interaction_action", {})
                for action_key in ("click_element", "input_text", "select_option"):
                    if action_key in ia and ia[action_key] is not None:
                        ia[action_key]["command"] = winning
                        logger.info(f"Node {idx}: promoted fallback locator → {winning}")
                        break

        elif outcome == "agentic":
            # Agentic succeeded — try to promote to deterministic using the step cache.
            # Read the actions the agentic actually took and build a command from the
            # first meaningful action (skip navigate/evaluate/send_keys-only entries).
            cached_acts = step_caches.get(idx, [])
            promoted = False
            for act in cached_acts:
                atype = act.get("action_type", "")
                if atype in ("navigate", "evaluate", "send_keys"):
                    continue  # skip side-effects
                cmd = _action_to_command(act)
                if not cmd:
                    continue
                ia = node.get("interaction_action", {})
                # If the node is still agentic_task (previous fallback), rebuild
                # it as the matching interaction type
                if "agentic_task" in ia:
                    if atype in ("click",):
                        node["interaction_action"] = {
                            "click_element": {
                                "command": cmd,
                                "prompt_instructions": ia["agentic_task"].get("task", "")[:80],
                                "skip_command": False,
                                "skip_prompt": False,
                                "assert_locator_presence": False,
                                "double_click": False,
                                "expect_download": False,
                                "button": "left",
                                "mouse_click": False,
                                "force": False,
                            }
                        }
                    elif atype in ("input", "type"):
                        text = act.get("action_params", {}).get("text", "")
                        # Substitute back to parameter ref if value matches
                        for key, values in input_parameters.items():
                            if isinstance(values, list) and text in values:
                                text = f"{{{key}[{values.index(text)}]}}"
                                break
                        node["interaction_action"] = {
                            "input_text": {
                                "command": cmd,
                                "input_text": text,
                                "fill_or_type": "fill",
                                "prompt_instructions": ia["agentic_task"].get("task", "")[:80],
                                "skip_command": False,
                                "skip_prompt": False,
                                "assert_locator_presence": False,
                                "click_before_input": True,
                                "press_enter": False,
                                "is_slider": False,
                            }
                        }
                    else:
                        continue
                    logger.info(f"Node {idx}: agentic → promoted to deterministic [{atype}] {cmd[:50]}")
                    promoted = True
                    break
                # If node already has a typed action, just update its command
                for action_key in ("click_element", "input_text", "select_option"):
                    if action_key in ia and ia[action_key] is not None:
                        ia[action_key]["command"] = cmd
                        logger.info(f"Node {idx}: agentic → updated command → {cmd[:50]}")
                        promoted = True
                        break
                if promoted:
                    break

            if not promoted:
                logger.debug(f"Node {idx}: agentic but no promotable cache found — leaving as-is.")

        elif outcome == "failed":
            original_hint = ""
            ia = node.get("interaction_action", {})
            for action_key in ("click_element", "input_text", "select_option"):
                action_data = ia.get(action_key)
                if action_data:
                    original_hint = action_data.get("prompt_instructions", "")
                    break

            task_description = original_hint or agentic_task_text
            node["interaction_action"] = {
                "agentic_task": {
                    "task": task_description,
                    "max_steps": 5,
                    "backend": "browser_use",
                }
            }
            logger.info(f"Node {idx}: failed → reverted to agentic_task")

    return patched


# ---------------------------------------------------------------------------
# Frontier detection — skip proven nodes, start from first failure
# ---------------------------------------------------------------------------

def find_frontier(
    outcomes: list[dict],
    task_dir: Path,
    auto_start_url: str,
) -> tuple[int, str]:
    """Return (first_unproven_node_idx, url_at_that_point).

    Scans proven nodes (command_success / deterministic / skipped) from the front.
    Once the first non-proven node is found, returns its index plus the URL the
    browser was at just before it ran (derived from the last proven step's cache).

    This lets the next round's automation skip repeating proven steps and start
    directly at the frontier, saving significant round time.
    """
    proven = {"command_success", "deterministic", "skipped"}
    sorted_outcomes = sorted(outcomes, key=lambda o: o["node"])

    # Walk forward through consecutively-proven nodes
    first_unproven = 0
    for o in sorted_outcomes:
        if o["outcome"] in proven:
            first_unproven = o["node"] + 1
        else:
            break  # first gap — stop here

    if first_unproven == 0:
        return 0, auto_start_url  # first node itself is unproven — can't skip anything

    # Find the URL the browser was at when the last proven node completed.
    # Walk backwards through step caches to find the most recent URL.
    url = auto_start_url
    for idx in range(first_unproven - 1, -1, -1):
        cache_path = task_dir / "logs" / f"step_{idx}" / "action_cache.json"
        if not cache_path.exists():
            continue
        try:
            data = json.loads(cache_path.read_text(errors="replace"))
            for act in reversed(data.get("deterministic_actions_list", [])):
                u = act.get("url", "")
                if u.startswith("http"):
                    url = u
                    logger.info(f"Frontier: node {first_unproven}, resume URL from step_{idx}: {url}")
                    return first_unproven, url
        except Exception:
            pass

    logger.info(f"Frontier: node {first_unproven}, no URL found — using start URL.")
    return first_unproven, url


# ---------------------------------------------------------------------------
# Cache merging helper
# ---------------------------------------------------------------------------

def _sort_step_caches(paths) -> list:
    """Sort step_N/action_cache.json paths by numeric N, not lexicographic."""
    def step_num(p):
        import re
        m = re.search(r"step_(\d+)", str(p))
        return int(m.group(1)) if m else -1
    return sorted(paths, key=step_num)


def find_latest_cache(task_dir: Path) -> Path | None:
    """Return the last step's action_cache.json (the trailing agentic's discoveries).

    For round 0 we also merge ALL step caches (it's the initial clean discovery).
    For rounds > 0 we ONLY want the trailing agentic's new discoveries — not the
    agentic-fallback caches from intermediate nodes which contain redundant/duplicated
    sequences that confuse the LLM builder.
    """
    step_caches = _sort_step_caches(task_dir.glob("logs/step_*/action_cache.json"))
    if not step_caches:
        top = task_dir / "logs" / "action_cache.json"
        return top if top.exists() else None

    # Return the last (numerically highest step_N) step's cache
    return step_caches[-1]


def find_merged_cache(task_dir: Path) -> Path | None:
    """Merge ALL per-step action caches (for round 0 initial full build)."""
    step_caches = _sort_step_caches(task_dir.glob("logs/step_*/action_cache.json"))
    if not step_caches:
        top = task_dir / "logs" / "action_cache.json"
        return top if top.exists() else None

    all_actions: list[dict] = []
    start_url = ""
    task_description = ""
    for cache_path in step_caches:
        try:
            with open(cache_path) as f:
                data = json.load(f)
            if not start_url:
                start_url = data.get("start_url", "")
            if not task_description:
                task_description = data.get("task_description", "")
            all_actions.extend(data.get("deterministic_actions_list", []))
        except Exception:
            pass

    if not all_actions:
        return None

    merged = {
        "task_description": task_description,
        "start_url": start_url,
        "total_actions": len(all_actions),
        "successful_actions": len(all_actions),
        "deterministic_actions": len(all_actions),
        "deterministic_actions_list": all_actions,
        "actions": all_actions,
    }

    merged_path = task_dir / "logs" / "merged_cache.json"
    merged_path.write_text(json.dumps(merged, indent=2))
    logger.info(f"Merged {len(all_actions)} actions from {len(step_caches)} step caches → {merged_path}")
    return merged_path


# ---------------------------------------------------------------------------
# Round 0: build initial 2-node agentic automation
# ---------------------------------------------------------------------------

def _enrich_task_with_params(task: str, input_parameters: dict) -> str:
    """Append the resolved input parameter values to any agentic task description.

    Without this, a vague instruction like "find the cheapest hotel and book it"
    gives the agent no concrete values to fill in — it will invent a destination,
    pick arbitrary dates, and use placeholder names.  Appending the actual values
    from input_parameters gives the agent a ground truth it must use when it
    encounters form fields, date pickers, or name inputs on the page.

    The output looks like:
        <original task>

        Use these exact values when filling in any form fields:
          destination city: Gurgaon
          check in date: Sunday, 05 October
          ...

    Works for any website and any parameter set — nothing is hardcoded here.
    """
    if not input_parameters:
        return task
    lines = []
    for key, values in input_parameters.items():
        if values:
            # Convert underscore keys to readable labels (e.g. check_in_date → check in date)
            label = key.replace("_", " ")
            lines.append(f"  {label}: {values[0]}")
    if not lines:
        return task
    return task + "\n\nUse these exact values when filling in any form fields:\n" + "\n".join(lines)


def make_agentic_automation(
    start_url: str,
    agentic_task: str,
    input_parameters: dict,
) -> dict:
    # Inject param values into the task so the round-0 agent uses the correct
    # destination, dates, and personal details instead of inventing its own.
    task_with_params = _enrich_task_with_params(agentic_task, input_parameters)
    return {
        "url": start_url,
        "parameters": {
            "input_parameters": input_parameters,
            "generated_parameters": {},
        },
        "nodes": [
            {
                "type": "action_node",
                "interaction_action": {
                    "close_overlay_popup": {
                        "task": "Close any popup or overlay. Look for X, close, or dismiss buttons and donot signin",
                        "max_steps": 3,
                        "backend": "browser_use",
                    }
                },
            },
            {
                "type": "action_node",
                "interaction_action": {
                    "agentic_task": {
                        "task": task_with_params,
                        "max_steps": 30,
                        "backend": "browser_use",
                    }
                },
            },
        ],
    }


# ---------------------------------------------------------------------------
# Task completion evaluator
# ---------------------------------------------------------------------------

def _get_final_url(task_dir: Path) -> str:
    """Read the URL the browser was on when the last node finished.

    Walks step dirs in reverse order looking for a state.json with a url field.
    Falls back to empty string if nothing found.
    """
    step_dirs = sorted(
        task_dir.glob("logs/step_*"),
        key=lambda p: int(re.search(r"step_(\d+)", str(p)).group(1)) if re.search(r"step_(\d+)", str(p)) else -1,
        reverse=True,
    )
    for step_dir in step_dirs:
        state_path = step_dir / "state.json"
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(errors="replace"))
                url = state.get("url", "")
                if url.startswith("http"):
                    return url
            except Exception:
                pass
    return ""


def _read_top_locator_score(task_dir: Path, node_idx: int) -> int | None:
    """Return the top stability score from logs/step_N/locator_candidates.json.

    Returns None if the file doesn't exist (non-locator action like scroll/go_to_url).
    Returns 0 if the file exists but can't be parsed.
    Score semantics: 0=JS eval failed, 10=xpath-only, >10=semantic attribute found.
    """
    candidates_path = task_dir / "logs" / f"step_{node_idx}" / "locator_candidates.json"
    if not candidates_path.exists():
        return None  # non-locator action — no element lookup involved
    try:
        candidates = json.loads(candidates_path.read_text(errors="replace"))
        if candidates:
            return int(candidates[0]["score"])  # list is already sorted best-first
    except Exception:
        pass
    return 0  # file exists but unreadable → treat as worst case


def _evaluate_task_completion(
    original_task: str,
    automation: dict,
    outcomes: list[dict],
    task_dir: Path,
    trailing_cache_path: Path | None,
) -> str | None:
    """Ask the LLM what sub-goals from the original task were NOT completed.

    Builds evidence from three sources:
    1. What the deterministic automation nodes were supposed to do
       (their prompt_instructions tell us what each node covers).
    2. What the trailing agentic agent actually did
       (its action cache lists the actions it took).
    3. The final URL the browser reached.

    Returns a short string of missed goals (one per line), or None if
    everything was achieved. The caller appends this to the next round's
    trailing agentic task so it explicitly fixes the gaps.
    """
    # --- Collect what the deterministic nodes cover ---
    # Each node's prompt_instructions describes its intent. Gathering them
    # gives the LLM a picture of what was already handled deterministically.
    covered_steps = []
    for node in automation.get("nodes", []):
        ia = node.get("interaction_action", {})
        for atype in ("click_element", "input_text", "select_option", "scroll", "go_to_url"):
            action = ia.get(atype)
            if isinstance(action, dict):
                hint = action.get("prompt_instructions", "")
                if hint:
                    covered_steps.append(hint)
                break

    # --- Collect what the trailing agentic actually did ---
    # The trailing agentic's action cache shows every action it took,
    # giving us concrete evidence of what it explored vs skipped.
    agentic_actions = []
    if trailing_cache_path and trailing_cache_path.exists():
        try:
            cache_data = json.loads(trailing_cache_path.read_text(errors="replace"))
            for act in cache_data.get("deterministic_actions_list", [])[:15]:
                atype = act.get("action_type", "")
                ax = (act.get("element") or {}).get("ax_name", "")
                text = act.get("action_params", {}).get("text", "")
                desc = f"{atype}"
                if ax:
                    desc += f" on '{ax}'"
                if text:
                    desc += f" → '{text[:40]}'"
                agentic_actions.append(desc)
        except Exception:
            pass

    final_url = _get_final_url(task_dir)
    summary = outcomes_summary(outcomes)

    # --- Build evaluation prompt ---
    covered_str = "\n".join(f"  - {s}" for s in covered_steps) if covered_steps else "  (none)"
    agentic_str = "\n".join(f"  - {a}" for a in agentic_actions) if agentic_actions else "  (none — agent did nothing)"

    prompt = f"""A browser automation ran with this goal:
"{original_task}"

What the deterministic automation nodes covered:
{covered_str}

What the trailing agentic agent did this round:
{agentic_str}

Final URL reached: {final_url}
Node outcomes: {summary}

List ONLY the specific sub-goals from the original task that were clearly NOT completed.
One line per missed goal. Be concrete and brief (e.g. "sort results by price before selecting hotel").
If all goals were achieved, respond with exactly: COMPLETE"""

    try:
        import litellm, os, time as _time
        model = os.environ.get("LLM_MODEL", "openai/gpt-4.1")
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

        # Wait for the token-per-minute window to partially reset before calling.
        # The trailing agentic often exhausts the TPM budget; a brief pause lets
        # the rate-limit counter roll over so this evaluation call succeeds.
        _time.sleep(15)

        resp = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": "You evaluate browser automation task completion. Be concise."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=300,
            api_key=api_key,
        )
        result = (resp.choices[0].message.content or "").strip()
        if result.upper() == "COMPLETE" or not result:
            logger.info("Task completion check: all goals achieved.")
            return None
        logger.info(f"Task completion check — missed goals:\n{result}")
        return result
    except Exception as e:
        # Non-critical — if evaluation fails, proceed without enrichment.
        # Log the full error type so we can diagnose rate limits or API errors.
        import traceback
        logger.warning(f"Task completion evaluation failed ({type(e).__name__}): {e}")
        logger.debug(f"Task completion evaluation traceback:\n{traceback.format_exc()}")
        return None


# ---------------------------------------------------------------------------
# Server restart helper
# ---------------------------------------------------------------------------

def restart_server(automation_path: str, env_path: str = ".env") -> None:
    """Kill the current optexity server and restart with a new automation override."""
    subprocess.run(["pkill", "-f", "optexity inference"], capture_output=True)
    time.sleep(2)
    env = {
        **os.environ,
        "OPTEXITY_LOCAL_AUTOMATION": str(automation_path),
        "ENV_PATH": env_path,
    }
    subprocess.Popen(
        ["conda", "run", "-n", "optexity", "optexity", "inference",
         "--port", "9000", "--child_process_id", "0"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for server to be ready
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=3)
            if r.ok:
                logger.info("Server ready.")
                return
        except Exception:
            pass
        time.sleep(1)
    logger.warning("Server may not be ready yet — continuing anyway.")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_loop(
    endpoint: str,
    input_parameters: dict,
    start_url: str,
    agentic_task: str,
    max_rounds: int = 5,
    env_path: str = ".env",
    output_dir: str = ".",
    resume_from_round: int | None = None,
) -> dict | None:
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    current_automation: dict | None = None
    current_automation_path: Path | None = None

    # Support resuming from a previously-built round N automation.
    # Set current_automation from the file and skip past the pure-agentic round 0.
    start_round = 0
    if resume_from_round is not None and resume_from_round > 0:
        seed_path = output_dir / f"automation_round_{resume_from_round}.json"
        if seed_path.exists():
            with open(seed_path) as f:
                current_automation = json.load(f)
            current_automation_path = seed_path
            start_round = resume_from_round
            logger.info(f"Resuming from round {resume_from_round}: {seed_path}")
        else:
            logger.warning(f"Resume path {seed_path} not found — starting from round 0.")

    for round_num in range(start_round, start_round + max_rounds):
        logger.info(f"\n{'='*60}")
        logger.info(f"ROUND {round_num}")
        logger.info(f"{'='*60}")

        # --- Build automation for this round ---
        if round_num == 0:
            auto = make_agentic_automation(start_url, agentic_task, input_parameters)
            auto_path = output_dir / "automation_round_0.json"
            with open(auto_path, "w") as f:
                json.dump(auto, f, indent=2)
            logger.info(f"Round 0: running agentic automation → {auto_path}")
        else:
            auto_path = current_automation_path
            logger.info(f"Round {round_num}: using {auto_path}")

        # --- Restart server with this automation ---
        restart_server(str(auto_path), env_path)

        # --- Trigger task ---
        task_id = trigger_task(endpoint, input_parameters)
        if task_id is None:
            logger.error("Could not trigger task — aborting loop.")
            break

        result = wait_for_task(task_id)
        if result is None:
            logger.error(f"Round {round_num}: task timed out.")
            break

        task_dir = find_task_dir(task_id)
        if task_dir is None:
            logger.error(f"Round {round_num}: task directory not found for {task_id}.")
            break

        # --- Read outcomes ---
        outcomes = read_node_outcomes(task_dir)
        summary = outcomes_summary(outcomes)
        logger.info(f"Round {round_num} outcomes: {summary}")

        log_path = task_dir / "logs" / "optexity.log"
        history_path = output_dir / "loop_history.json"

        # --- Find action cache ---
        # Round 0: merge all step caches (clean discovery — use everything)
        # Round > 0: only trailing agentic's step cache (its new discoveries)
        if round_num == 0:
            cache_path = find_merged_cache(task_dir)
        else:
            cache_path = find_latest_cache(task_dir)

        if cache_path is None:
            logger.warning(f"Round {round_num}: no action cache found — skipping build.")
            break

        logger.info(f"Round {round_num}: cache at {cache_path}")

        # Count how many new deterministic actions the trailing cache has
        with open(cache_path) as f:
            _cache_data = json.load(f)
        new_det_actions = len(_cache_data.get("deterministic_actions_list", []))
        logger.info(f"Round {round_num}: {new_det_actions} deterministic actions in trailing cache.")

        # --- Check convergence ---
        # Converge when the trailing agentic found 0 new actions AND task succeeded.
        # We do NOT require command_success on every node — prompt_fallback nodes
        # reliably work via LLM locator and would never reach command_success, causing
        # an infinite loop if we insisted on it.
        
        trailing_agentic_done = new_det_actions == 0 and result.get("status") == "success"
        if round_num > 0 and trailing_agentic_done:
            logger.info(f"Converged at round {round_num}: trailing agentic idle, task succeeded.")
            return current_automation

        # --- Collect failed-node history from this round ---
        failed_errors = extract_failed_errors(log_path)
        history = build_failed_history(round_num, current_automation or {}, outcomes, failed_errors, history_path)
        failed_history_str = format_failed_history(history) or None
        if failed_history_str:
            logger.info(f"Failed history ({len(history)} entries):\n{failed_history_str}")

        # --- Build next automation ---
        from llm_cache_to_automation import llm_convert, llm_improve

        if round_num == 0:
            # Round 0: LLM builds the initial automation from the full discovery cache
            try:
                next_auto = llm_convert(cache_path, input_parameters, start_url, failed_history=failed_history_str)
            except Exception as e:
                logger.warning(f"LLM builder failed ({e}), using rule-based fallback.")
                from cache_to_automation import convert_cache_to_automation
                next_auto = convert_cache_to_automation(cache_path, input_parameters, start_url)
        else:
            # Round > 0: outcome-driven LLM improvement.
            # The LLM sees exactly what happened to each non-deterministic node
            # (error messages, winning locators, agentic actions) and fixes them
            # directly — no hardcoded rules needed.

            proven = {"command_success", "deterministic", "skipped"}
            fallback_locators = extract_fallback_locators(log_path)
            failed_errors = extract_failed_errors(log_path)

            # The trailing agentic is always the last node; exclude it from evidence
            # so llm_improve doesn't try to convert its actions into new nodes (which
            # causes duplicates). Its actions are handled by the convergence check only.
            trailing_node_idx = len(current_automation.get("nodes", [])) - 1

            # Build per-node evidence for non-deterministic nodes (skip trailing agentic)
            node_evidence: list[dict] = []
            for o in outcomes:
                if o["outcome"] in proven:
                    continue
                if o["node"] == trailing_node_idx:
                    continue  # trailing agentic — handled separately
                evidence: dict = {
                    "node": o["node"],
                    "outcome": o["outcome"],
                    "error": failed_errors.get(o["node"]),
                    "winning_locator": fallback_locators.get(o["node"]),
                    "winning_actions": [],
                }
                sc_path = task_dir / "logs" / f"step_{o['node']}" / "action_cache.json"
                if sc_path.exists():
                    try:
                        sc_data = json.loads(sc_path.read_text(errors="replace"))
                        evidence["winning_actions"] = sc_data.get("deterministic_actions_list", [])
                    except Exception:
                        pass
                node_evidence.append(evidence)

            if node_evidence:
                logger.info(
                    f"Round {round_num}: LLM improving {len(node_evidence)} non-deterministic nodes: "
                    + str([f"n{e['node']}={e['outcome']}" for e in node_evidence])
                )

            # Always improve the full automation starting from the original URL.
            # Proven nodes keep their state; the LLM only touches non-deterministic ones.
            next_auto = llm_improve(current_automation, node_evidence)

            # Remove the existing trailing agentic (last agentic node at end)
            nodes = next_auto.get("nodes", [])
            while nodes and "agentic_task" in nodes[-1].get("interaction_action", {}):
                nodes.pop()

            # Deduplicate nodes: if two nodes share the same (action_type, command),
            # keep the first occurrence only. This prevents llm_improve from producing
            # duplicate SEARCH / Escape nodes when the LLM ignores the "no duplicate" rule.
            seen_keys: set[tuple] = set()
            deduped: list[dict] = []
            for n in nodes:
                ia_n = n.get("interaction_action", {})
                action_key = None
                for atype in ("click_element", "input_text", "scroll_element", "select_option"):
                    if atype in ia_n:
                        cmd = ia_n[atype].get("command", "")
                        action_key = (atype, cmd)
                        break
                if action_key is None or action_key not in seen_keys:
                    deduped.append(n)
                    if action_key is not None:
                        seen_keys.add(action_key)
                else:
                    logger.info(f"Deduped duplicate node: {action_key}")
            nodes = deduped

            next_auto["nodes"] = nodes

        # --- Enforce max_tries on all nodes ---
        # Proven nodes get max_tries=10 only if their locator is stable (score>10).
        # score=0: locator.evaluate() failed, stability unknown.
        # score=10: xpath-only, breaks on any DOM reorder.
        # score>10: semantic attribute (id, aria-label, placeholder…), safe to promote.
        # score=None: no locator_candidates.json → non-locator action (scroll, go_to_url…)
        #             → always stable, promote freely.
        proven_set = {"command_success", "deterministic", "skipped"}
        proven_indices = {o["node"] for o in outcomes if o["outcome"] in proven_set}
        outcome_by_node = {o["node"]: o["outcome"] for o in outcomes}
        for node_idx, node in enumerate(next_auto.get("nodes", [])):
            ia = node.get("interaction_action", {})
            if "agentic_task" in ia:
                continue
            if node_idx in proven_indices:
                top_score = _read_top_locator_score(task_dir, node_idx)
                # command_success should always produce a candidates file.
                # If it's missing, locator_from_playwright was caught by handle_command's
                # outer guard — no stability signal, so don't promote.
                if top_score is None and outcome_by_node.get(node_idx) == "command_success":
                    top_score = 0
                ia["max_tries"] = 10 if (top_score is None or top_score > 10) else 2
            else:
                ia["max_tries"] = 2

        # --- Evaluate task quality: did the automation achieve all sub-goals? ---
        # After each round (not just mechanical failures), ask the LLM whether
        # every sub-goal of the original vague task was actually carried out.
        # If anything was missed (e.g. "sort by price" skipped), the missed goals
        # are appended to the next trailing agentic task so the agent self-corrects
        # without the user having to spell out every step upfront.
        #
        # trailing_cache_path is the last agentic step's cache — it shows what
        # the trailing agent did, so the evaluator can compare intent vs. action.
        trailing_cache_path = find_latest_cache(task_dir)
        missed_goals = _evaluate_task_completion(
            original_task=agentic_task,
            automation=next_auto,
            outcomes=outcomes,
            task_dir=task_dir,
            trailing_cache_path=trailing_cache_path,
        )

        # Build the effective task for the next round's trailing agentic.
        #
        # Layer 1 — always: inject the resolved input parameter values so the agent
        # knows exactly which city, dates, and personal details to use.  Without this
        # the agent invents values every run (e.g. "New Delhi" instead of "Gurgaon").
        # _enrich_task_with_params is generalized — it works for any param set.
        task_with_params = _enrich_task_with_params(agentic_task, input_parameters)

        # Layer 2 — when the quality evaluator found missed sub-goals: append them
        # so the agent explicitly retries what it skipped last round (e.g. "sort by price").
        # The original task is preserved; everything here is additive.
        if missed_goals:
            enriched_task = (
                f"{task_with_params}\n\n"
                f"Previous run missed these goals — make sure to complete them:\n"
                f"{missed_goals}"
            )
            logger.info(f"Task enriched for next round with missed goals:\n{missed_goals}")

            # Write evolution log so the user can see what the system learned each round
            evolution_path = output_dir / "task_evolution.json"
            try:
                evolution = json.loads(evolution_path.read_text()) if evolution_path.exists() else []
            except Exception:
                evolution = []
            evolution.append({"round": round_num, "missed_goals": missed_goals.splitlines()})
            with open(evolution_path, "w") as f:
                json.dump(evolution, f, indent=2)
        else:
            enriched_task = task_with_params

        # --- Append trailing agentic node ---
        # Uses enriched_task (params always injected + missed goals if any)
        # so the agent has concrete values and explicit retry instructions.
        trailing_node = {
            "type": "action_node",
            "interaction_action": {
                "agentic_task": {
                    "task": enriched_task,
                    "max_steps": 20,
                    "backend": "browser_use",
                }
            },
        }
        next_auto["nodes"].append(trailing_node)
        logger.info(f"Appended trailing agentic node (node {len(next_auto['nodes'])-1}).")

        # --- Validate ---
        from optexity.schema.automation import Automation
        try:
            Automation.model_validate(next_auto)
            n_nodes = len(next_auto.get("nodes", []))
            logger.info(f"Round {round_num}: built {n_nodes}-node automation, schema valid.")
        except Exception as e:
            logger.error(f"Round {round_num}: automation schema invalid — {e}")
            break

        # --- Save ---
        next_path = output_dir / f"automation_round_{round_num + 1}.json"
        with open(next_path, "w") as f:
            json.dump(next_auto, f, indent=2)
        logger.info(f"Saved → {next_path}")

        current_automation = next_auto
        current_automation_path = next_path

    logger.info("Loop ended.")
    return current_automation


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Smart iterative automation loop")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--params", required=True, help="JSON file with input_parameters")
    parser.add_argument("--url", required=True, help="Start URL")
    parser.add_argument("--agentic-task", required=True, help="Task description for agentic nodes")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--env-path", default=".env")
    parser.add_argument("--output-dir", default="./loop_output")
    parser.add_argument("--resume-from-round", type=int, default=None,
                        help="Resume loop starting from automation_round_N.json")
    args = parser.parse_args()

    with open(args.params) as f:
        p = json.load(f)
    input_parameters = (
        p.get("input_parameters")
        or p.get("parameters", {}).get("input_parameters", {})
        or p
    )

    final = run_loop(
        endpoint=args.endpoint,
        input_parameters=input_parameters,
        start_url=args.url,
        agentic_task=args.agentic_task,
        max_rounds=args.max_rounds,
        env_path=args.env_path,
        output_dir=args.output_dir,
        resume_from_round=args.resume_from_round,
    )

    if final:
        out = Path(args.output_dir) / "automation_final.json"
        with open(out, "w") as f:
            json.dump(final, f, indent=2)
        print(f"\nFinal automation saved → {out}")
    else:
        print("\nLoop did not converge or failed.")


if __name__ == "__main__":
    main()
