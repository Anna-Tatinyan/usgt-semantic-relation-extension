import json
from typing import List, Dict


VALID_STORY_LABELS = {"DUPLICATE", "DEPENDS_ON", "CONFLICT"}
VALID_ACTION_LABELS = {"BLOCKS", "ENABLES"}
SYMMETRIC_STORY_LABELS = {"DUPLICATE", "CONFLICT"}


def load_prediction_file(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_action_id(story_text: str, action_name: str) -> str:
    return f"{story_text}::ACTION::{action_name}"


def add_inter_story_relations(graph, predicted_rows: List[Dict]):
    for row in predicted_rows:
        label = row["label"].upper().strip()

        if label not in VALID_STORY_LABELS:
            continue

        if label in SYMMETRIC_STORY_LABELS:
            query = f"""
            MATCH (s1:UserStory {{text: $source_story}})
            MATCH (s2:UserStory {{text: $target_story}})
            MERGE (s1)-[r1:{label}]->(s2)
            SET r1.predicted = true
            MERGE (s2)-[r2:{label}]->(s1)
            SET r2.predicted = true
            """
        else:
            query = f"""
            MATCH (s1:UserStory {{text: $source_story}})
            MATCH (s2:UserStory {{text: $target_story}})
            MERGE (s1)-[r:{label}]->(s2)
            SET r.predicted = true
            """

        graph.query(
            query,
            params={
                "source_story": row["source_story"],
                "target_story": row["target_story"],
            },
        )


def add_inter_story_relations(graph, predicted_rows: List[Dict]):
    for row in predicted_rows:
        label = row["label"].upper().strip()

        if label not in VALID_ACTION_LABELS:
            continue

        story_text = row["story_id"]
        action_1_id = make_action_id(story_text, row["action_1"])
        action_2_id = make_action_id(story_text, row["action_2"])

        query = f"""
        MATCH (s:UserStory {{text: $story_text}})-[:HAS_ACTION]->(a1:Action {{id: $action_1_id}})
        MATCH (s)-[:HAS_ACTION]->(a2:Action {{id: $action_2_id}})
        MERGE (a1)-[r:{label}]->(a2)
        SET r.predicted = true
        """

        graph.query(
            query,
            params={
                "story_text": story_text,
                "action_1_id": action_1_id,
                "action_2_id": action_2_id,
            },
        )