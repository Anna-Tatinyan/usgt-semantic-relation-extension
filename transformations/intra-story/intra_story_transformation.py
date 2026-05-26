import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR.parent.parent / ".env"

load_dotenv(ENV_PATH)

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise EnvironmentError(
        f"OPENAI_API_KEY is missing. Put it in: {ENV_PATH}"
    )

PROMPT_DIR = BASE_DIR / "prompts"
SEED_DIR = BASE_DIR / "seeds"

OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

client = OpenAI(api_key=OPENAI_API_KEY)


# Approximate manual cost tracking only.
# Check the OpenAI pricing page before final reporting.
PRICE_PER_1M = {
    "gpt-5.5": {"input": 5.00, "output": 30.00},
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
}


TRANSFORMATION_FILES = {
    "PRECONDITION_ACTION_TRANSFORMATION": "precondition.txt",
    "PREVENTING_CONDITION_ACTION_TRANSFORMATION": "preventing.txt",
}


# ============================================================
# Run selection
# ============================================================
# If both are True: 66 stories × 2 transformations = 132 jobs.
# If only one is True: 66 jobs.

RUN_FAMILIES = {
    "PRECONDITION": False,
    "PREVENTING": True,
}

# Change this when running only one family, e.g. "precondition_only".
RUN_NAME = "intra_story"


SEED_FILE = "g14_stories_for_prevent.json"
EXPECTED_NUM_STORIES = 66


OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "story_id": {"type": "string"},
        "source_story": {"type": "string"},
        "rewritten_story": {"type": "string"},
        "transformation": {"type": "string"},
        "action_1": {"type": "string"},
        "action_2": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": [
        "story_id",
        "source_story",
        "rewritten_story",
        "transformation",
        "action_1",
        "action_2",
        "rationale",
    ],
}


# ============================================================
# File helpers
# ============================================================

def load_text(filename: str) -> str:
    path = PROMPT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing prompt file: {path}")
    return path.read_text(encoding="utf-8")


def load_json(filename: str) -> List[Dict[str, str]]:
    path = SEED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing seed file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clean_prompt_template(text: str) -> str:
    placeholder = "Input stories:\n[PASTE SEED STORIES HERE]"
    return text.replace(placeholder, "").strip()


def save_job_manifest(jobs: List[Dict[str, Any]]) -> None:
    manifest_path = OUT_DIR / f"{RUN_NAME}_job_manifest.csv"

    rows = []

    for index, job in enumerate(jobs, start=1):
        rows.append({
            "job_index": index,
            "source_set": job["source_set"],
            "batch": job["batch"],
            "story_id": job["story_id"],
            "transformation": job["transformation"],
            "source_story": job["source_story"],
        })

    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "job_index",
                "source_set",
                "batch",
                "story_id",
                "transformation",
                "source_story",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Job manifest saved to: {manifest_path}")


def save_results_table(
        items: List[Dict[str, Any]],
        metadata: List[Dict[str, Any]],
) -> None:
    results_path = OUT_DIR / f"{RUN_NAME}_transformations_result.csv"

    rows = []

    for item, meta in zip(items, metadata):
        usage = meta.get("usage", {}) or {}

        rows.append({
            "job_index": meta.get("job_index"),
            "source_set": meta.get("source_set"),
            "batch": meta.get("batch"),
            "story_id": item.get("story_id"),
            "transformation": item.get("transformation"),
            "source_story": item.get("source_story"),
            "rewritten_story": item.get("rewritten_story"),
            "action_1": item.get("action_1"),
            "action_2": item.get("action_2"),
            "rationale": item.get("rationale"),

            # Time tracking
            "started_at": meta.get("started_at"),
            "finished_at": meta.get("finished_at"),
            "duration_seconds": meta.get("duration_seconds"),

            # Token tracking
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),

            # Cost tracking
            "estimated_cost_usd": meta.get("estimated_cost_usd"),

            "validation_errors": json.dumps(
                meta.get("validation_errors", []),
                ensure_ascii=False,
            ),
        })

    with results_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "job_index",
                "source_set",
                "batch",
                "story_id",
                "transformation",
                "source_story",
                "rewritten_story",
                "action_1",
                "action_2",
                "rationale",
                "started_at",
                "finished_at",
                "duration_seconds",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "estimated_cost_usd",
                "validation_errors",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results table saved to: {results_path}")


def check_for_duplicate_jobs(jobs: List[Dict[str, Any]]) -> None:
    seen = {}
    duplicates = []

    for index, job in enumerate(jobs, start=1):
        key = (
            job["source_set"],
            job["story_id"],
            job["transformation"],
        )

        if key in seen:
            duplicates.append({
                "first_job_index": seen[key],
                "duplicate_job_index": index,
                "source_set": job["source_set"],
                "story_id": job["story_id"],
                "transformation": job["transformation"],
                "batch": job["batch"],
            })
        else:
            seen[key] = index

    if duplicates:
        duplicate_path = OUT_DIR / f"{RUN_NAME}_duplicate_job_warnings.json"
        save_json(duplicate_path, duplicates)

        print("Duplicate jobs found:")
        for dup in duplicates:
            print(
                f"  source_set={dup['source_set']} | "
                f"story_id={dup['story_id']} | "
                f"transformation={dup['transformation']} | "
                f"first={dup['first_job_index']} | "
                f"duplicate={dup['duplicate_job_index']}"
            )

        raise ValueError(
            f"Found {len(duplicates)} duplicate source/story/transformation jobs. "
            f"Details saved to {duplicate_path}"
        )

    print("Duplicate check OK: no repeated source_set + story_id + transformation pairs.")


# ============================================================
# Prompt construction
# ============================================================

def build_system_prompt(transformation: str) -> str:
    general_prompt = clean_prompt_template(load_text("general.txt"))
    transformation_prompt = clean_prompt_template(
        load_text(TRANSFORMATION_FILES[transformation])
    )

    return f"""
{general_prompt}

{transformation_prompt}

Additional execution rules:
- You will receive exactly one input story.
- Return exactly one JSON object, not an array wrapper.
- Preserve the input story_id exactly.
- Preserve the input source_story exactly.
- The transformation field must be exactly: {transformation}
- The rewritten_story field must contain the transformed user story.
- The action_1 and action_2 fields must identify the two intended main actions.
- Do not include markdown.
- Do not include code fences.
- Do not include explanatory text outside the JSON object.
""".strip()


def build_user_prompt(story: Dict[str, str]) -> str:
    return "Input story:\n" + json.dumps(story, ensure_ascii=False, indent=2)


# ============================================================
# Job construction
# ============================================================

def make_jobs() -> List[Dict[str, Any]]:
    stories = load_json(SEED_FILE)

    if len(stories) != EXPECTED_NUM_STORIES:
        raise ValueError(
            f"Expected {EXPECTED_NUM_STORIES} source stories, got {len(stories)}"
        )

    jobs = []

    def add_job(
            source_set: str,
            batch: str,
            transformation: str,
            story: Dict[str, str],
    ) -> None:
        jobs.append({
            "source_set": source_set,
            "batch": batch,
            "transformation": transformation,
            "story": story,
            "story_id": str(story["story_id"]),
            "source_story": story["source_story"],
        })

    if RUN_FAMILIES["PRECONDITION"]:
        for story in stories:
            add_job(
                source_set="g14",
                batch="precondition_action",
                transformation="PRECONDITION_ACTION_TRANSFORMATION",
                story=story,
            )

    if RUN_FAMILIES["PREVENTING"]:
        for story in stories:
            add_job(
                source_set="g14",
                batch="preventing_condition_action",
                transformation="PREVENTING_CONDITION_ACTION_TRANSFORMATION",
                story=story,
            )

    if not jobs:
        raise ValueError("No jobs selected. Set at least one RUN_FAMILIES value to True.")

    check_for_duplicate_jobs(jobs)
    save_job_manifest(jobs)

    print("Selected run families:")
    for family, enabled in RUN_FAMILIES.items():
        print(f"  {family}: {enabled}")

    print(f"Total selected jobs: {len(jobs)}")

    return jobs


# ============================================================
# Cost and validation
# ============================================================

def extract_usage(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)

    if usage is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "raw_usage": None,
        }

    usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)

    input_tokens = usage_dict.get("input_tokens", 0)
    output_tokens = usage_dict.get("output_tokens", 0)
    total_tokens = usage_dict.get("total_tokens", input_tokens + output_tokens)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "raw_usage": usage_dict,
    }


def estimate_cost(usage: Dict[str, Any], model: str) -> float | None:
    price = PRICE_PER_1M.get(model)

    if price is None:
        return None

    input_cost = usage["input_tokens"] / 1_000_000 * price["input"]
    output_cost = usage["output_tokens"] / 1_000_000 * price["output"]

    return input_cost + output_cost


def validate_item(
        item: Dict[str, Any],
        original_story: Dict[str, str],
        expected_transformation: str,
) -> List[str]:
    errors = []

    expected_id = str(original_story["story_id"])
    expected_source = original_story["source_story"]

    if str(item.get("story_id")) != expected_id:
        errors.append(
            f"story_id mismatch: expected {expected_id}, got {item.get('story_id')}"
        )

    if item.get("source_story") != expected_source:
        errors.append("source_story was changed")

    if item.get("transformation") != expected_transformation:
        errors.append(
            f"transformation mismatch: expected {expected_transformation}, "
            f"got {item.get('transformation')}"
        )

    if not item.get("rewritten_story"):
        errors.append("rewritten_story is empty")

    if not item.get("action_1"):
        errors.append("action_1 is empty")

    if not item.get("action_2"):
        errors.append("action_2 is empty")

    if not item.get("rationale"):
        errors.append("rationale is empty")

    if item.get("rewritten_story") == expected_source:
        errors.append("rewritten_story is identical to source_story")

    return errors


# ============================================================
# API call
# ============================================================

def call_openai_for_one_story(
        story: Dict[str, str],
        transformation: str,
        model: str = MODEL,
        max_retries: int = 3,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    system_prompt = build_system_prompt(transformation)
    user_prompt = build_user_prompt(story)

    call_started_at = datetime.now(timezone.utc)
    call_start_perf = time.perf_counter()

    last_exception = None

    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                reasoning={
                    "effort": "medium"
                },
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "single_intra_story_transformation",
                        "strict": True,
                        "schema": OUTPUT_SCHEMA,
                    }
                },
                max_output_tokens=1200,
            )

            output_text = response.output_text
            item = json.loads(output_text)

            usage = extract_usage(response)
            estimated_cost = estimate_cost(usage, model)

            call_finished_at = datetime.now(timezone.utc)
            call_duration_seconds = time.perf_counter() - call_start_perf

            metadata = {
                "model": model,
                "transformation": transformation,
                "story_id": str(story["story_id"]),
                "started_at": call_started_at.isoformat(),
                "finished_at": call_finished_at.isoformat(),
                "duration_seconds": round(call_duration_seconds, 3),
                "usage": usage,
                "estimated_cost_usd": estimated_cost,
                "attempt": attempt + 1,
            }

            return item, metadata

        except Exception as e:
            last_exception = e
            wait_seconds = 2 * (attempt + 1)
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)

    raise RuntimeError(
        f"API call failed after {max_retries} attempts "
        f"for story_id={story.get('story_id')}"
    ) from last_exception


# ============================================================
# Main
# ============================================================

def main() -> None:
    run_started_at = datetime.now(timezone.utc)
    run_start_perf = time.perf_counter()

    jobs = make_jobs()

    # Test mode:
    # Uncomment while testing. Comment again for the full selected run.
    # jobs = jobs[:2]

    all_items = []
    all_metadata = []
    all_validation_errors = []

    print(f"Model: {MODEL}")
    print(f"Run name: {RUN_NAME}")
    print(f"Total jobs to run: {len(jobs)}")
    print(f"Output folder: {OUT_DIR.resolve()}")
    print()

    for index, job in enumerate(jobs, start=1):
        source_set = job["source_set"]
        batch = job["batch"]
        transformation = job["transformation"]
        story = job["story"]
        story_id = job["story_id"]

        print(
            f"[{index}/{len(jobs)}] "
            f"source_set={source_set} | "
            f"story_id={story_id} | "
            f"batch={batch} | "
            f"transformation={transformation}"
        )

        item, metadata = call_openai_for_one_story(
            story=story,
            transformation=transformation,
            model=MODEL,
        )

        validation_errors = validate_item(
            item=item,
            original_story=story,
            expected_transformation=transformation,
        )

        metadata["source_set"] = source_set
        metadata["batch"] = batch
        metadata["job_index"] = index
        metadata["validation_errors"] = validation_errors

        if validation_errors:
            print(f"  Validation errors: {validation_errors}")
        else:
            print("  Validation OK")

        all_items.append(item)
        all_metadata.append(metadata)

        # Incremental CSV/JSON backup after every completed item.
        save_results_table(all_items, all_metadata)
        save_json(OUT_DIR / f"{RUN_NAME}_partial_combined.json", {"items": all_items})
        save_json(OUT_DIR / f"{RUN_NAME}_partial_metadata.json", all_metadata)

        if metadata.get("validation_errors"):
            all_validation_errors.append({
                "job_index": index,
                "source_set": source_set,
                "batch": batch,
                "story_id": story_id,
                "transformation": transformation,
                "errors": metadata["validation_errors"],
            })

        cost = metadata.get("estimated_cost_usd")
        if cost is not None:
            print(f"  Estimated cost: ${cost:.6f}")
        else:
            print("  Estimated cost: unknown")

        print()

    combined_output = {
        "items": all_items
    }

    run_finished_at = datetime.now(timezone.utc)
    run_duration_seconds = time.perf_counter() - run_start_perf

    run_summary = {
        "run_name": RUN_NAME,
        "model": MODEL,
        "selected_families": RUN_FAMILIES,
        "seed_file": SEED_FILE,
        "total_jobs": len(jobs),
        "total_items": len(all_items),
        "total_validation_error_jobs": len(all_validation_errors),
        "validation_errors": all_validation_errors,
        "total_input_tokens": sum(
            m.get("usage", {}).get("input_tokens", 0) for m in all_metadata
        ),
        "total_output_tokens": sum(
            m.get("usage", {}).get("output_tokens", 0) for m in all_metadata
        ),
        "total_tokens": sum(
            m.get("usage", {}).get("total_tokens", 0) for m in all_metadata
        ),
        "estimated_total_cost_usd": sum(
            m.get("estimated_cost_usd", 0) or 0 for m in all_metadata
        ),
        "started_at": run_started_at.isoformat(),
        "finished_at": run_finished_at.isoformat(),
        "duration_seconds": round(run_duration_seconds, 3),
        "duration_minutes": round(run_duration_seconds / 60, 3),
    }

    combined_path = OUT_DIR / f"{RUN_NAME}_combined.json"
    metadata_path = OUT_DIR / f"{RUN_NAME}_metadata.json"
    summary_path = OUT_DIR / f"{RUN_NAME}_summary.json"
    results_table_path = OUT_DIR / f"{RUN_NAME}_transformations_result.csv"
    manifest_path = OUT_DIR / f"{RUN_NAME}_job_manifest.csv"

    save_json(combined_path, combined_output)
    save_json(metadata_path, all_metadata)
    save_json(summary_path, run_summary)
    save_results_table(all_items, all_metadata)

    print("Finished.")
    print(f"Combined output: {combined_path}")
    print(f"Run metadata: {metadata_path}")
    print(f"Run summary: {summary_path}")
    print(f"Job manifest: {manifest_path}")
    print(f"Results table: {results_table_path}")
    print()
    print(f"Total input tokens: {run_summary['total_input_tokens']}")
    print(f"Total output tokens: {run_summary['total_output_tokens']}")
    print(f"Total tokens: {run_summary['total_tokens']}")
    print(f"Estimated total cost: ${run_summary['estimated_total_cost_usd']:.4f}")
    print(f"Validation error jobs: {run_summary['total_validation_error_jobs']}")
    print(f"Runtime seconds: {run_summary['duration_seconds']}")
    print(f"Runtime minutes: {run_summary['duration_minutes']}")


if __name__ == "__main__":
    main()