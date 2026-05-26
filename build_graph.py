# # build_graph.py
# import os
# from dotenv import load_dotenv, find_dotenv
#
# from extractor import extract_backlog_graphs, get_graph
# from predicted_relations import (
#     load_prediction_file,
#     add_inter_story_relations,
#     add_inter_story_relations,
# )
#
#
# def clear_database(graph):
#     graph.query("MATCH (n) DETACH DELETE n")
#
#
# def build_full_graph(
#         backlog_numbers,
#         experiment: str,
#         inter_story_predictions_path: str = None,
#         intra_story_predictions_path: str = None,
#         clear_first: bool = False,
# ):
#     load_dotenv(find_dotenv(), override=True)
#     graph = get_graph()
#
#     if clear_first:
#         print("Clearing Neo4j database...")
#         clear_database(graph)
#
#     print("Step 1: Extracting and pushing base graph...")
#     for backlog_number in backlog_numbers:
#         print(f"Processing backlog {backlog_number}")
#         extract_backlog_graphs(
#             backlog_number=backlog_number,
#             experiment=experiment,
#             save_files=True,
#             push_to_neo4j=True,
#         )
#
#     print("Step 2: Adding predicted inter-story relations...")
#     if inter_story_predictions_path and os.path.exists(inter_story_predictions_path):
#         inter_story_rows = load_prediction_file(inter_story_predictions_path)
#         add_inter_story_relations(graph, inter_story_rows)
#     else:
#         print("No inter-story prediction file found. Skipping.")
#
#     print("Step 3: Adding predicted intra-story relations...")
#     if intra_story_predictions_path and os.path.exists(intra_story_predictions_path):
#         intra_story_rows = load_prediction_file(intra_story_predictions_path)
#         add_inter_story_relations(graph, intra_story_rows)
#     else:
#         print("No intra-story prediction file found. Skipping.")
#
#     print("Full enriched graph build completed.")
#
#
# if __name__ == "__main__":
#     build_full_graph(
#         backlog_numbers=["14"],
#         experiment="relationExtract",
#         inter_story_predictions_path="relationship_predictions/predicted-relation-extension-result/applicability_study/g14_inter_story_study_turbo.json",
#         intra_story_predictions_path="relationship_predictions/predicted-relation-extension-result/applicability_study/g14_intra_story_study_turbo.json",
#         clear_first=True,
#     )