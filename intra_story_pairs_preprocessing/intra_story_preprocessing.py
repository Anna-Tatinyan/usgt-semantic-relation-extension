import os
import json
from itertools import combinations

backlog_number = "14"

print(f"Processing backlog {backlog_number}")

input_path = os.path.join(
    "../annotated_ground_truth",
    f"g{backlog_number}.json"
)

with open(input_path, "r", encoding="utf-8") as file:
    stories = json.load(file)


def normalize_pid(pid):
    """
    Converts values like:
        g14  -> #G14#
        G14  -> #G14#
        #G14# -> #G14#
    """
    if not isinstance(pid, str) or not pid.strip():
        return None

    pid = pid.strip().replace("#", "").upper()
    return f"#{pid}#"


def normalize_text(text, pid):
    """
    Ensures text starts with the normalized PID.
    """
    if not isinstance(text, str):
        text = ""

    text = text.strip()
    normalized_pid = normalize_pid(pid)

    if normalized_pid and not text.startswith(normalized_pid):
        text = f"{normalized_pid} {text}"

    return text


def normalize_action(action):
    if not isinstance(action, str):
        return None

    action = action.strip()

    if not action:
        return None

    return " ".join(action.split())


def extract_actions(story):
    """
    Extracts actions from:

        story["Action"]["Primary Action"]
        story["Action"]["Secondary Action"]

    Returns:
        actions_clean: unique cleaned actions in original order
        issues: list of warning strings
    """
    issues = []

    action_block = story.get("Action", {})

    if not isinstance(action_block, dict):
        return [], ["malformed Action field"]

    raw_primary = action_block.get("Primary Action", [])
    raw_secondary = action_block.get("Secondary Action", [])

    if not isinstance(raw_primary, list):
        issues.append("Primary Action is not a list")
        raw_primary = []

    if not isinstance(raw_secondary, list):
        issues.append("Secondary Action is not a list")
        raw_secondary = []

    raw_actions = raw_primary + raw_secondary

    cleaned = []
    seen = set()
    duplicates = set()
    malformed_count = 0

    for action in raw_actions:
        normalized = normalize_action(action)

        if normalized is None:
            malformed_count += 1
            continue

        if normalized in seen:
            duplicates.add(normalized)
        else:
            seen.add(normalized)
            cleaned.append(normalized)

    if malformed_count:
        issues.append(f"{malformed_count} empty/malformed action(s) removed")

    if duplicates:
        issues.append(f"exact duplicate action(s): {sorted(duplicates)}")

    if not cleaned:
        issues.append("no valid actions found")

    return cleaned, issues


def build_action_pairs_per_story(data):
    results = []

    for i, story in enumerate(data):
        raw_pid = story.get("PID", "")
        pid = normalize_pid(raw_pid)

        text = normalize_text(story.get("Text", ""), raw_pid)

        actions, issues = extract_actions(story)

        action_pairs = [list(pair) for pair in combinations(actions, 2)]

        if len(actions) < 2:
            issues.append("fewer than 2 valid actions, so no pairs generated")

        results.append({
            "story_index": i,
            "PID": pid,
            "Text": text,
            "actions": actions,
            "action_pairs": action_pairs,
            "issues": issues
        })

    return results


results = build_action_pairs_per_story(stories)

output_path = f"g{backlog_number}_action_pairs.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4, ensure_ascii=False)

print(f"Saved to {output_path}")