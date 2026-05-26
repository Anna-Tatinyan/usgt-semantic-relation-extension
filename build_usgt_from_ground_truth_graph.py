import os
import json
import re
from dotenv import load_dotenv, find_dotenv
from langchain_community.graphs import Neo4jGraph

from predicted_relations import (
    load_prediction_file,
    add_inter_story_relations,
    add_inter_story_relations,
)


def get_graph():
    load_dotenv(find_dotenv(), override=True)
    return Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE"),
    )


def clear_database(graph):
    graph.query("MATCH (n) DETACH DELETE n")


def normalize_persona(value):
    if not isinstance(value, str):
        return None
    value = " ".join(value.strip().split())
    if not value:
        return None
    return value.lower()


def normalize_value(value):
    if not isinstance(value, str):
        return None
    value = " ".join(value.strip().split())
    if not value:
        return None
    return value


def clean_story_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"#G\d{2}#", "", text).strip()


def make_action_id(story_text, action_name):
    return f"{story_text}::ACTION::{action_name}"


def unique_clean(values, normalizer=normalize_value):
    cleaned = []
    seen = set()

    if not isinstance(values, list):
        return cleaned

    for v in values:
        norm = normalizer(v)
        if norm is None:
            continue
        if norm not in seen:
            seen.add(norm)
            cleaned.append(norm)

    return cleaned


def unique_clean_pairs(pairs, source_normalizer=normalize_value, target_normalizer=normalize_value):
    cleaned = []
    seen = set()

    if not isinstance(pairs, list):
        return cleaned

    for pair in pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            continue

        s = source_normalizer(pair[0])
        t = target_normalizer(pair[1])

        if s is None or t is None:
            continue

        key = (s, t)
        if key not in seen:
            seen.add(key)
            cleaned.append([s, t])

    return cleaned


def build_story_graph_from_annotation(graph, story, story_index):
    raw_text = story.get("Text", "")
    story_text = clean_story_text(raw_text)

    if not story_text:
        return

    personas = unique_clean(story.get("Persona", []), normalizer=normalize_persona)

    action_block = story.get("Action", {})
    primary_actions = unique_clean(action_block.get("Primary Action", []))
    secondary_actions = unique_clean(action_block.get("Secondary Action", []))
    actions = unique_clean(primary_actions + secondary_actions)

    entity_block = story.get("Entity", {})
    primary_entities = unique_clean(entity_block.get("Primary Entity", []))
    secondary_entities = unique_clean(entity_block.get("Secondary Entity", []))
    entities = unique_clean(primary_entities + secondary_entities)

    benefit = normalize_value(story.get("Benefit", ""))

    triggers = unique_clean_pairs(
        story.get("Triggers", []),
        source_normalizer=normalize_persona,
        target_normalizer=normalize_value,
    )
    targets = unique_clean_pairs(
        story.get("Targets", []),
        source_normalizer=normalize_value,
        target_normalizer=normalize_value,
    )
    contains = unique_clean_pairs(
        story.get("Contains", []),
        source_normalizer=normalize_value,
        target_normalizer=normalize_value,
    )

    graph.query(
        """
        MERGE (s:UserStory {text: $story_text})
        SET s.story_index = $story_index
        """,
        params={
            "story_text": story_text,
            "story_index": story_index,
        },
    )

    for persona in personas:
        graph.query(
            """
            MATCH (s:UserStory {text: $story_text})
            MERGE (p:Persona {name: $persona})
            MERGE (s)-[:HAS_PERSONA]->(p)
            """,
            params={
                "story_text": story_text,
                "persona": persona,
            },
        )

    for action in actions:
        action_id = make_action_id(story_text, action)
        graph.query(
            """
            MATCH (s:UserStory {text: $story_text})
            MERGE (a:Action {id: $action_id})
            SET a.name = $action_name
            MERGE (s)-[:HAS_ACTION]->(a)
            """,
            params={
                "story_text": story_text,
                "action_id": action_id,
                "action_name": action,
            },
        )

    for entity in entities:
        graph.query(
            """
            MATCH (s:UserStory {text: $story_text})
            MERGE (e:Entity {name: $entity})
            MERGE (s)-[:HAS_ENTITY]->(e)
            """,
            params={
                "story_text": story_text,
                "entity": entity,
            },
        )

    if benefit:
        graph.query(
            """
            MATCH (s:UserStory {text: $story_text})
            MERGE (b:Benefit {text: $benefit})
            MERGE (s)-[:HAS_BENEFIT]->(b)
            """,
            params={
                "story_text": story_text,
                "benefit": benefit,
            },
        )

    for persona, action in triggers:
        action_id = make_action_id(story_text, action)
        graph.query(
            """
            MATCH (p:Persona {name: $persona})
            MATCH (s:UserStory {text: $story_text})-[:HAS_ACTION]->(a:Action {id: $action_id})
            MERGE (p)-[:TRIGGERS]->(a)
            """,
            params={
                "story_text": story_text,
                "persona": persona,
                "action_id": action_id,
            },
        )

    for action, entity in targets:
        action_id = make_action_id(story_text, action)
        graph.query(
            """
            MATCH (s:UserStory {text: $story_text})-[:HAS_ACTION]->(a:Action {id: $action_id})
            MATCH (s)-[:HAS_ENTITY]->(e:Entity {name: $entity})
            MERGE (a)-[:TARGETS]->(e)
            """,
            params={
                "story_text": story_text,
                "action_id": action_id,
                "entity": entity,
            },
        )

    for source_entity, target_entity in contains:
        graph.query(
            """
            MATCH (s:UserStory {text: $story_text})-[:HAS_ENTITY]->(e1:Entity {name: $source_entity})
            MATCH (s)-[:HAS_ENTITY]->(e2:Entity {name: $target_entity})
            MERGE (e1)-[:CONTAINS]->(e2)
            """,
            params={
                "story_text": story_text,
                "source_entity": source_entity,
                "target_entity": target_entity,
            },
        )


def load_ground_truth(backlog_number):
    path = os.path.join("data/thesis_annotations/g14_subset_applicability_study.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_base_graph_from_ground_truth(graph, backlog_number):
    stories = load_ground_truth(backlog_number)

    for idx, story in enumerate(stories):
        build_story_graph_from_annotation(graph, story, idx)

    print(f"Base graph built directly from ground_truth/g{backlog_number}.json")


def build_full_graph_from_ground_truth(
        backlog_number,
        inter_story_predictions_path=None,
        action_predictions_path=None,
        clear_first=False,
):
    graph = get_graph()

    if clear_first:
        print("Clearing Neo4j database...")
        clear_database(graph)

    print("Step 1: Building base graph from ground-truth annotations...")
    build_base_graph_from_ground_truth(graph, backlog_number)

    print("Step 2: Adding predicted inter-story relations...")
    if inter_story_predictions_path and os.path.exists(inter_story_predictions_path):
        inter_story_rows = load_prediction_file(inter_story_predictions_path)
        add_inter_story_relations(graph, inter_story_rows)
    else:
        print("No inter-story prediction file found. Skipping.")

    print("Step 3: Adding predicted intra-story action relations...")
    if action_predictions_path and os.path.exists(action_predictions_path):
        action_rows = load_prediction_file(action_predictions_path)
        add_inter_story_relations(graph, action_rows)
    else:
        print("No action prediction file found. Skipping.")

    print("Full enriched graph build completed.")


if __name__ == "__main__":
    build_full_graph_from_ground_truth(
        backlog_number="14",
        inter_story_predictions_path="relationship_predictions/predicted-relation-extension-result/turbo/g14_inter_story_study_turbo.json",
        action_predictions_path="relationship_predictions/predicted-relation-extension-result/turbo/g14_intra_story_study_turbo.json",
        clear_first=True,
    )