import json
import os
import sys
import pickle

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def check_folder_exists(folder):
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError as e:
            print(f"Coudn't create folder {folder} because of {e}")
            sys.exit()
    
    return True


def save_graph_document_to_json(graph_documents, pid, experiment):
    """
    Save the graph documents to a json file. If the file already exists, append the new data to the existing data.

    Args:
        graph_documents (List[GraphDocument]): A list of graph documents
        pid (str): The backlog ID from POS_BASELINE
        experiment (str): The experiment name

    Returns:
        List[Dict]: A list of dictionaries containing the extracted user stories
    """
    graph_documents_dicts = [graph_document.dict() for graph_document in graph_documents]
    outputs = []

    for graph_document_dict in graph_documents_dicts:
        output = load_json("template.json")

        # Fill in the fields based on the result
        output["ID"] = ''
        output["PID"] = pid
        output["Text"] = graph_document_dict["source"]["page_content"]

        for node in graph_document_dict["nodes"]:
            if node["type"] == "Persona":
                output["Persona"].append(node["id"])
            elif node["type"] == "Entity":
                output["Entity"]["Primary Entity"].append(node["id"])
            elif node["type"] == "Action":
                output["Action"]["Primary Action"].append(node["id"])
            elif node["type"] == "Benefit":
                output["Benefit"] = node["id"]

        for relationship in graph_document_dict["relationships"]:
            if relationship["type"] == "TRIGGERS":
                output["Triggers"].append([relationship["source"]["id"], relationship["target"]["id"]])
            if relationship["type"] == "TARGETS":
                output["Targets"].append([relationship["source"]["id"], relationship["target"]["id"]])

        outputs.append(output)

        # Check if file exists
    file_path = 'usgt_extracted_baseline/'+experiment+'/'+pid+'.json'
    if os.path.exists(file_path):
        # If file exists, load existing data
        with open(file_path, 'r') as f:
            existing_data = json.load(f)
        # Append new data to existing data
        existing_data.extend(outputs)
        outputs = existing_data
    else:
        # If file does not exist, create new file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save to json
    with open(file_path, 'w') as f:
        json.dump(outputs, f, indent=4)


    return outputs

def save_graph_document_to_pickle(graph_documents, pid, experiment):
    """
    Save the graph documents to a pickle file. If the file already exists, append the new data to the existing data.

    Args:
        graph_documents (List[GraphDocument]): A list of graph documents
        pid (str): The backlog ID from POS_BASELINE
        experiment (str): The experiment name

    Returns:
        List[Dict]: A list of GraphDocuments containing the extracted user stories
    """

    save_path = 'usgt_extracted_baseline/'+experiment+'/'+'pickle/'
    folder = check_folder_exists(save_path)

    if folder: 
        file_path = save_path+pid+'.pickle'
        if os.path.exists(file_path):
            # Load existing data
            with open(file_path, 'rb') as file:
                existing_data = pickle.load(file)
        else:
            existing_data = []

        # Append new data
        existing_data.extend(graph_documents)

        # Write combined data back to file
        with open(file_path, 'wb') as file:
            pickle.dump(existing_data, file)

    return existing_data


