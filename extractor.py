# extractor.py
import os
import json
from dotenv import load_dotenv, find_dotenv

from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langchain_community.graphs import Neo4jGraph

from preprocess import preprocess_text
from save_results import (
    save_graph_document_to_json,
    save_graph_document_to_pickle,
)
from us_graph_transformer import UserStoryGraphTransformer


def get_llm():
    load_dotenv(find_dotenv(), override=True)
    return OllamaLLM(model=os.getenv("OLLAMA_MODEL", "llama3"), temperature=0)


def get_graph():
    load_dotenv(find_dotenv(), override=True)
    return Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE"),
    )


def get_transformer():
    llm = get_llm()
    return UserStoryGraphTransformer(llm=llm)


def load_backlog(backlog_number: str):
    backlog_path = os.path.join(
        "data/usgt_extracted_baseline/gpt-4-turbo/subset/",
        f"g{backlog_number}.json"
    )
    with open(backlog_path, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_story_graph(item, transformer, graph=None, experiment=None, save_files=True, push_to_neo4j=True):
    story_text = item["Text"]
    story, pid = preprocess_text(story_text)
    doc = [Document(page_content=story)]

    graph_document = transformer.convert_to_graph_documents(doc)

    if save_files and experiment is not None:
        save_graph_document_to_json(graph_document, pid, experiment)
        save_graph_document_to_pickle(graph_document, pid, experiment)

    if push_to_neo4j and graph is not None:
        graph.add_graph_documents(graph_document)

    return graph_document, pid


def extract_backlog_graphs(
        backlog_number: str,
        experiment: str,
        save_files: bool = True,
        push_to_neo4j: bool = True,
):
    transformer = get_transformer()
    graph = get_graph() if push_to_neo4j else None

    if save_files:
        out_dir = os.path.join("annotated_ground_truth", experiment)
        os.makedirs(out_dir, exist_ok=True)

    backlog = load_backlog(backlog_number)
    graph_documents = []

    total = len(backlog)
    count = 0

    for item in backlog:
        try:
            graph_document, pid = extract_story_graph(
                item=item,
                transformer=transformer,
                graph=graph,
                experiment=experiment,
                save_files=save_files,
                push_to_neo4j=push_to_neo4j,
            )
            graph_documents.extend(graph_document)
            count += 1
            print(f"{count}/{total} stories processed (pid={pid})")
        except Exception as e:
            print(f"User story error: {e}")
            continue

    return graph_documents


if __name__ == "__main__":
    experiment = "relationExtract"
    backlog_list = ["02"]

    for backlog_number in backlog_list:
        print(f"Processing backlog {backlog_number}")
        extract_backlog_graphs(
            backlog_number=backlog_number,
            experiment=experiment,
            save_files=True,
            push_to_neo4j=True,
        )