import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, cast
import re
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, create_model

examples = [
    {
        "text": "As a business owner, I want to give my inputs on the product development.",
        "head": "business owner",
        "head_type": "Persona",
        "relation": "TRIGGERS",
        "tail": "give",
        "tail_type": "Action",
    },
    {
        "text": "As a business owner, I want to give inputs on the product development.",
        "head": "give",
        "head_type": "Action",
        "relation": "TARGETS",
        "tail": "inputs",
        "tail_type": "Entity",
    },
    {
        "text": "As a business owner, I want to give inputs on the product development.",
        "head": "give",
        "head_type": "Action",
        "relation": "TARGETS",
        "tail": "product development",
        "tail_type": "Entity",
    },
    {
        "text": "As an analyst, I want to move on to the next phase, so that I can have updated reports.",
        "head": "move on",
        "head_type": "Action",
        "relation": "TARGETS",
        "tail": "next phase",
        "tail_type": "Entity",
    },
    {
        "text": "As an analyst, I want to move on to the next phase, so that I can have updated reports.",
        "head": "have",
        "head_type": "Action",
        "relation": "TARGETS",
        "tail": "updated reports",
        "tail_type": "Entity",
    },
]

system_prompt = (
    """
Knowledge Graph Constructor Instructions \n
## 1. Overview \n
You are a specialized requirements engineer, who understands about scrum framework. Your task is to analyze and extract nodes and relationships from user stories to build a knowledge graph. 
You have to extract as much information as possible without sacrificing accuracy. Do not add any information that is not explicitly in the mentioned user story. \n
## 2. Nodes \n
Nodes represent concepts in a user story. Given a user story, you need to extract: \n
    - Persona: there is only one persona node per user story, introduced as 'As a *persona*,'. \n
    - Actions: are all verbs in the user story that describe what the persona desires to do (e.g. move on, access, have). Extract the verb only, without modifiers.\n
    - Entities: are nouns and each noun must be extracted as a separate entity, even if they seem related or grouped. Include any modifiers that clarify the entity (e.g. library database, domain).  \n
**Consistency**: Ensure you use available types for node labels, you necessarily extract at least 4 nodes: persona, action, entity.\n
**Node IDs**: Never utilize integers as node IDs. Node IDs should be names or human-readable identifiers extracted as found in the user story.\n
**Extract all actions and entities**: capture every action and its corresponding entity.\n
**Separate verbs**: consider each verb as a distinct action and its objects as related entities.\n
## 3. Relationships\n
Relationships represent connections between nodes. The only possible relationships are:\n
    - Persona->main action (triggers). \n
    - Action->entity (targets). \n
No other relationships are allowed except for the ones above, make sure to create all the possible relationships. \n
## 4. Coreference Resolution \n
**Maintain Entity Consistency**: When extracting entities, it's vital to ensure consistency.\n
If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"), 
always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID.\n'
Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.\n
## 5. Strict Compliance\n
Adhere to the rules strictly. Non-compliance will result in termination.
## 6. Example \n
'As a user, I want to sync my data so that I can access my information from anywhere.' \n
Extracted Nodes: \n
Persona: ['user'] \n
Action: ['sync', 'access'] \n
Entity: ['data', 'current information', 'anywhere'] \n
Relationships: \n
TRIGGERS: [['user', 'sync']] \n
TARGETS: [['sync', 'data'], ['access', 'current information']] \n
"""
)

default_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        (
            "human",
            (
                "Tip: Make sure to answer in the correct format and do "
                "not include any explanations. "
                "Use the given format to extract information from the "
                "following input: {input}"
            ),
        ),
    ]
)

benefit_prompt = (
    """
    You are a specialized requirements engineer, that understand about scrum framework.\n
    You have to extract as much information as possible without sacrificing accuracy. 
    Do not add any information that is not explicitly in the mentioned user story.\n
    ## Benefit\n
    Extract the benefit sentence of the user story, if it exists.
    The benefit sentence is a sentence typically introduced as 'so that *benefit*', 'in order to *benefit*'.
    ## Examples\n
    if benefit sentence exists: \n  
    input: 'As a user, I want to sync my data, so that I can access my information from anywhere.'\n
    answer: Node(id='I can access my information from anywhere', type='Benefit')\n
    if benefit sentence does not exist: \n
    input: 'As a customer, I want to pay by cash.' \n
    answer: '' \n
    """
)


benefit_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            benefit_prompt,
        ),
        (
            "human",
            (
                "Tip: Make sure to answer in the correct format and do "
                "not include any explanations. "
                "Use the given format to extract information from the "
                "following input: {input}"
            ),
        ),
    ]
)



def _get_additional_info(input_type: str) -> str:
    # Check if the input_type is one of the allowed values
    if input_type not in ["node", "relationship", "property"]:
        raise ValueError("input_type must be 'node', 'relationship', or 'property'")

    # Perform actions based on the input_type
    if input_type == "node":
        return (
            "Ensure you use basic or elementary types for node labels.\n"
            "For example, when you identify an entity representing a person, "
            "always label it as **'Persona'**. Avoid using more specific terms "
            "like 'Mathematician' or 'Scientist'"
        )
    elif input_type == "relationship":
        return (
            "The relationships can be TRIGGERS (from Persona to Action) or TARGETS (from Action to Entity)."
        )
    elif input_type == "property":
        return ""
    return ""


def optional_enum_field(
    enum_values: Optional[List[str]] = None,
    description: str = "",
    input_type: str = "node",
    llm_type: Optional[str] = None,
    **field_kwargs: Any,
) -> Any:
    """Utility function to conditionally create a field with an enum constraint."""
    # Only openai supports enum param
    if enum_values and llm_type == "openai-chat":
        return Field(
            ...,
            enum=enum_values,  # type: ignore[call-arg]
            description=f"{description}. Available options are {enum_values}",
            **field_kwargs,
        )
    elif enum_values:
        return Field(
            ...,
            description=f"{description}. Available options are {enum_values}",
            **field_kwargs,
        )
    else:
        additional_info = _get_additional_info(input_type)
        return Field(..., description=description + additional_info, **field_kwargs)


class _Graph(BaseModel):
    nodes: Optional[List]
    relationships: Optional[List]


class UnstructuredRelation(BaseModel):
    head: str = Field(
        description=(
            "extracted head entity like developer, dashboard, get. "
            "Must use human-readable unique identifier."
        )
    )
    head_type: str = Field(
        description="type of the extracted head entity like Persona, Entity, etc"
    )
    relation: str = Field(description="relation between the head and the tail entities")
    tail: str = Field(
        description=(
            "extracted tail entity like developer, dashboard, get. "
            "Must use human-readable unique identifier."
        )
    )
    tail_type: str = Field(
        description="type of the extracted tail entity like Persona, Entity, etc"
    )


def create_unstructured_prompt(
    node_labels: Optional[List[str]] = None, rel_types: Optional[List[str]] = None
) -> ChatPromptTemplate:
    node_labels_str = str(node_labels) if node_labels else ""
    rel_types_str = str(rel_types) if rel_types else ""
    base_string_parts = [
        """Knowledge Graph Constructor Instructions \n
## 1. Overview \n
You are a specialized requirements engineer, who understands about scrum framework. Your task is to analyze and extract nodes and relationships from user stories to build a knowledge graph. 
You have to extract as much information as possible without sacrificing accuracy. Do not add any information that is not explicitly in the mentioned user story. \n
## 2. Nodes \n
Nodes represent concepts in a user story. Given a user story, you need to extract: \n
    - Persona: there is only one persona node per user story, introduced as 'As a *persona*,'. \n
    - Actions: are all verbs in the user story that describe what the persona desires to do (e.g. move on, access, have). Extract the verb only, without modifiers.\n
    - Entities: are nouns and each noun must be extracted as a separate entity, even if they seem related or grouped. Include any modifiers that clarify the entity (e.g. library database, domain).  \n
**Consistency**: Ensure you use available types for node labels, you necessarily extract at least 4 nodes: persona, action, entity.\n
**Node IDs**: Never utilize integers as node IDs. Node IDs should be names or human-readable identifiers extracted as found in the user story.\n
**Extract all actions and entities**: capture every action and its corresponding entity.\n
**Separate verbs**: consider each verb as a distinct action and its objects as related entities.\n
## 3. Relationships\n
Relationships represent connections between nodes. The only possible relationships are:\n
    - Persona->main action (triggers). \n
    - Action->entity (targets). \n
No other relationships are allowed except for the ones above, make sure to create all the possible relationships. \n
## 4. Coreference Resolution \n
**Maintain Entity Consistency**: When extracting entities, it's vital to ensure consistency.\n
If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"), 
always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID.\n'
Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.\n
## 5. Strict Compliance\n
Adhere to the rules strictly. Non-compliance will result in termination.""",
        f'The "head_type" key must contain the type of the extracted head entity, '
        f"which must be one of the types from {node_labels_str}."
        if node_labels
        else "",
        f'The "relation" key must contain the type of relation between the "head" '
        f'and the "tail", which must be one of the relations from {rel_types_str}.'
        if rel_types
        else "",
        f'The "tail" key must represent the text of an extracted entity which is '
        f'the tail of the relation, and the "tail_type" key must contain the type '
        f"of the tail entity from {node_labels_str}."
        if node_labels
        else "",
        "Attempt to extract as many entities and relations as you can. Maintain "
        "Entity Consistency: When extracting entities, it's vital to ensure "
        'consistency. If an entity, such as "John Doe", is mentioned multiple '
        "times in the text but is referred to by different names or pronouns "
        '(e.g., "Joe", "he"), always use the most complete identifier for '
        "that entity. The knowledge graph should be coherent and easily "
        "understandable, so maintaining consistency in entity references is "
        "crucial.",
        "IMPORTANT NOTES:\n- Don't add any explanation and text.",
    ]
    system_prompt = "\n".join(filter(None, base_string_parts))

    system_message = SystemMessage(content=system_prompt)
    parser = JsonOutputParser(pydantic_object=UnstructuredRelation)

    human_string_parts = [
        "Based on the following example, extract entities and "
        "relations from the provided text.\n\n",
        "Use the following entity types, don't use other entity "
        "that is not defined below:"
        "# ENTITY TYPES:"
        "{node_labels}"
        if node_labels
        else "",
        "Use the following relation types, don't use other relation "
        "that is not defined below:"
        "# RELATION TYPES:"
        "{rel_types}"
        if rel_types
        else "",
        "Below are a number of examples of text and their extracted "
        "entities and relationships."
        "{examples}\n"
        "For the following text, extract entities and relations as "
        "in the provided example."
        "{format_instructions}\nText: {input}",
    ]
    human_prompt_string = "\n".join(filter(None, human_string_parts))
    human_prompt = PromptTemplate(
        template=human_prompt_string,
        input_variables=["input"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "node_labels": node_labels,
            "rel_types": rel_types,
            "examples": examples,
        },
    )

    human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message, human_message_prompt]
    )
    return chat_prompt


def create_simple_model(
    node_labels: Optional[List[str]] = None,
    rel_types: Optional[List[str]] = None,
    node_properties: Union[bool, List[str]] = False,
    llm_type: Optional[str] = None,
    relationship_properties: Union[bool, List[str]] = False,
) -> Type[_Graph]:
    """
    Create a simple graph model with optional constraints on node
    and relationship types.

    Args:
        node_labels (Optional[List[str]]): Specifies the allowed node types.
            Defaults to None, allowing all node types.
        rel_types (Optional[List[str]]): Specifies the allowed relationship types.
            Defaults to None, allowing all relationship types.
        node_properties (Union[bool, List[str]]): Specifies if node properties should
            be included. If a list is provided, only properties with keys in the list
            will be included. If True, all properties are included. Defaults to False.
        relationship_properties (Union[bool, List[str]]): Specifies if relationship
            properties should be included. If a list is provided, only properties with
            keys in the list will be included. If True, all properties are included.
            Defaults to False.
        llm_type (Optional[str]): The type of the language model. Defaults to None.
            Only openai supports enum param: openai-chat.

    Returns:
        Type[_Graph]: A graph model with the specified constraints.

    Raises:
        ValueError: If 'id' is included in the node or relationship properties list.
    """

    node_fields: Dict[str, Tuple[Any, Any]] = {
        "id": (
            str,
            Field(..., description="Name or human-readable unique identifier."),
        ),
        "type": (
            str,
            optional_enum_field(
                node_labels,
                description="The type or label of the node.",
                input_type="node",
                llm_type=llm_type,
            ),
        ),
    }

    if node_properties:
        if isinstance(node_properties, list) and "id" in node_properties:
            raise ValueError("The node property 'id' is reserved and cannot be used.")
        # Map True to empty array
        node_properties_mapped: List[str] = (
            [] if node_properties is True else node_properties
        )

        class Property(BaseModel):
            """A single property consisting of key and value"""

            key: str = optional_enum_field(
                node_properties_mapped,
                description="Property key.",
                input_type="property",
                llm_type=llm_type,
            )
            value: str = Field(..., description="value")

        node_fields["properties"] = (
            Optional[List[Property]],
            Field(None, description="List of node properties"),
        )
    SimpleNode = create_model("SimpleNode", **node_fields)  # type: ignore

    relationship_fields: Dict[str, Tuple[Any, Any]] = {
        "source_node_id": (
            str,
            Field(
                ...,
                description="Name or human-readable unique identifier of source node",
            ),
        ),
        "source_node_type": (
            str,
            optional_enum_field(
                node_labels,
                description="The type or label of the source node.",
                input_type="node",
                llm_type=llm_type,
            ),
        ),
        "target_node_id": (
            str,
            Field(
                ...,
                description="Name or human-readable unique identifier of target node",
            ),
        ),
        "target_node_type": (
            str,
            optional_enum_field(
                node_labels,
                description="The type or label of the target node.",
                input_type="node",
                llm_type=llm_type,
            ),
        ),
        "type": (
            str,
            optional_enum_field(
                rel_types,
                description="The type of the relationship.",
                input_type="relationship",
                llm_type=llm_type,
            ),
        ),
    }
    if relationship_properties:
        if (
            isinstance(relationship_properties, list)
            and "id" in relationship_properties
        ):
            raise ValueError(
                "The relationship property 'id' is reserved and cannot be used."
            )
        # Map True to empty array
        relationship_properties_mapped: List[str] = (
            [] if relationship_properties is True else relationship_properties
        )

        class RelationshipProperty(BaseModel):
            """A single property consisting of key and value"""

            key: str = optional_enum_field(
                relationship_properties_mapped,
                description="Property key.",
                input_type="property",
                llm_type=llm_type,
            )
            value: str = Field(..., description="value")

        relationship_fields["properties"] = (
            Optional[List[RelationshipProperty]],
            Field(None, description="List of relationship properties"),
        )
    SimpleRelationship = create_model("SimpleRelationship", **relationship_fields)  # type: ignore

    class DynamicGraph(_Graph):
        """Represents a graph document consisting of nodes and relationships."""

        nodes: Optional[List[SimpleNode]] = Field(description="List of nodes")  # type: ignore
        relationships: Optional[List[SimpleRelationship]] = Field(  # type: ignore
            description="List of relationships"
        )

        @classmethod
        def parse_nodes_and_relationships(cls, nodes: str, relationships: str):
            # Parse the nodes and relationships strings into lists of dictionaries
            nodes = json.loads(nodes)
            relationships = json.loads(relationships)

            # Convert the dictionaries into SimpleNode and SimpleRelationship instances
            nodes = [SimpleNode(**node) for node in nodes]
            relationships = [SimpleRelationship(**relationship) for relationship in relationships]

            # Create and return a new DynamicGraph instance
            return cls(nodes=nodes, relationships=relationships)

    return DynamicGraph


def normalize_persona_id(node_id: str) -> str:
    if not isinstance(node_id, str):
        return node_id
    return " ".join(node_id.strip().lower().split())

def normalize_node_id(node_id: str, node_type: str) -> str:
    if node_type == "Persona":
        return normalize_persona_id(node_id)
    return node_id


def map_to_base_node(node: Any) -> Node:
    properties = {}
    if hasattr(node, "properties") and node.properties:
        for p in node.properties:
            properties[format_property_key(p.key)] = p.value

    node_id = node.id
    if node.type == "Persona":
        node_id = normalize_persona_id(node_id)

    return Node(id=node_id, type=node.type, properties=properties)


def map_to_base_relationship(rel: Any) -> Relationship:
    source_id = rel.source_node_id
    target_id = rel.target_node_id

    if rel.source_node_type == "Persona":
        source_id = normalize_persona_id(source_id)
    if rel.target_node_type == "Persona":
        target_id = normalize_persona_id(target_id)

    source = Node(id=source_id, type=rel.source_node_type)
    target = Node(id=target_id, type=rel.target_node_type)

    properties = {}
    if hasattr(rel, "properties") and rel.properties:
        for p in rel.properties:
            properties[format_property_key(p.key)] = p.value

    return Relationship(
        source=source, target=target, type=rel.type, properties=properties
    )


def _parse_and_clean_json(
    argument_json: Dict[str, Any],
) -> Tuple[List[Node], List[Relationship]]:
    nodes = []
    for node in argument_json["nodes"]:
        if not node.get("id"):  # Id is mandatory, skip this node
            continue
        node_properties = {}
        if "properties" in node and node["properties"]:
            for p in node["properties"]:
                node_properties[format_property_key(p["key"])] = p["value"]
        nodes.append(
            Node(
                id=node["id"],
                type=node.get("type", "Node"),
                properties=node_properties,
            )
        )
    relationships = []
    for rel in argument_json["relationships"]:
        # Mandatory props
        if (
            not rel.get("source_node_id")
            or not rel.get("target_node_id")
            or not rel.get("type")
        ):
            continue

        # Node type copying if needed from node list
        if not rel.get("source_node_type"):
            try:
                rel["source_node_type"] = [
                    el.get("type")
                    for el in argument_json["nodes"]
                    if el["id"] == rel["source_node_id"]
                ][0]
            except IndexError:
                rel["source_node_type"] = None
        if not rel.get("target_node_type"):
            try:
                rel["target_node_type"] = [
                    el.get("type")
                    for el in argument_json["nodes"]
                    if el["id"] == rel["target_node_id"]
                ][0]
            except IndexError:
                rel["target_node_type"] = None

        rel_properties = {}
        if "properties" in rel and rel["properties"]:
            for p in rel["properties"]:
                rel_properties[format_property_key(p["key"])] = p["value"]

        source_node = Node(
            id=rel["source_node_id"],
            type=rel["source_node_type"],
        )
        target_node = Node(
            id=rel["target_node_id"],
            type=rel["target_node_type"],
        )
        relationships.append(
            Relationship(
                source=source_node,
                target=target_node,
                type=rel["type"],
                properties=rel_properties,
            )
        )
    return nodes, relationships


def _format_nodes(nodes: List[Node]) -> List[Node]:
    return [
        Node(
            id=el.id,
            type=el.type.capitalize()  # type: ignore[arg-type]
            if el.type
            else None,  # handle empty strings  # type: ignore[arg-type]
            properties=el.properties,
        )
        for el in nodes
    ]


def _format_relationships(rels: List[Relationship]) -> List[Relationship]:
    return [
        Relationship(
            source=_format_nodes([el.source])[0],
            target=_format_nodes([el.target])[0],
            type=el.type.replace(" ", "_").upper(),
            properties=el.properties,
        )
        for el in rels
    ]


def format_property_key(s: str) -> str:
    words = s.split()
    if not words:
        return s
    first_word = words[0].lower()
    capitalized_words = [word.capitalize() for word in words[1:]]
    return "".join([first_word] + capitalized_words)


def _convert_to_graph_document(
    raw_schema: Dict[Any, Any],
) -> Tuple[List[Node], List[Relationship]]:
    # If there are validation errors
    if not raw_schema["parsed"]:
        try:
            try:  # OpenAI type response
                argument_json = json.loads(
                    raw_schema["raw"].additional_kwargs["tool_calls"][0]["function"][
                        "arguments"
                    ]
                )
            except Exception:  # Google type response
                try:
                    argument_json = json.loads(
                        raw_schema["raw"].additional_kwargs["function_call"][
                            "arguments"
                        ]
                    )
                except Exception:  # Ollama type response
                    argument_json = raw_schema["raw"].tool_calls[0]["args"]
                    if isinstance(argument_json["nodes"], str):
                        argument_json["nodes"] = json.loads(argument_json["nodes"])
                    if isinstance(argument_json["relationships"], str):
                        argument_json["relationships"] = json.loads(
                            argument_json["relationships"]
                        )

            nodes, relationships = _parse_and_clean_json(argument_json)
        except Exception:  # If we can't parse JSON
            return ([], [])
    else:  # If there are no validation errors use parsed pydantic object
        parsed_schema: _Graph = raw_schema["parsed"]

        # add user story as a node
        user_story_node = Node(id=raw_schema["user_story"], type='Userstory')
        parsed_schema.nodes.append(user_story_node)

        #adds logical relationships
        parsed_schema.relationships += create_logical_rel(parsed_schema.nodes)


        nodes = (
            [map_to_base_node(node) for node in parsed_schema.nodes if node.id]
            if parsed_schema.nodes
            else []
        )

        relationships = (
            [
                map_to_base_relationship(rel)
                for rel in parsed_schema.relationships
                if rel.type and rel.source_node_id and rel.target_node_id
            ]
            if parsed_schema.relationships
            else []
        )
    # Title / Capitalize
    return nodes, _format_relationships(relationships)

def create_logical_rel(nodes) -> List[Relationship]:
    rels = []
    # find userstory node
    for node in nodes:
        if node.type == 'Userstory':
            user_story_node = node

    for node in nodes:
        if node.type == 'Persona':
            rels.append(create_relationship(user_story_node, node, 'HAS_PERSONA'))
        elif node.type == 'Action':
            rels.append(create_relationship(user_story_node, node, 'HAS_ACTION'))
        elif node.type == 'Entity':
            rels.append(create_relationship(user_story_node, node, 'HAS_ENTITY'))
        elif node.type == 'Benefit':
            rels.append(create_relationship(user_story_node, node, 'HAS_BENEFIT'))
    return rels

def create_relationship(source_node, target_node, rel_type):
    # Create a SimpleRelationship model class
    SimpleRelationship = create_simple_relationship_model(source_node_type=source_node.type, target_node_type=target_node.type, rel_type=rel_type)

    # Create a SimpleRelationship instance
    relationship = SimpleRelationship(source_node_id=source_node.id, source_node_type=source_node.type, target_node_id=target_node.id, target_node_type=target_node.type, type=rel_type)

    return relationship

def create_simple_relationship_model(
    source_node_type: Optional[str] = None,
    target_node_type: Optional[str] = None,
    rel_type: Optional[str] = None,
    rel_properties: Union[bool, List[str]] = False,
    llm_type: Optional[str] = None,
) -> Type[BaseModel]:
    """
    Create a simple relationship model with optional constraints on relationship type, source node type, target node type, and properties.

    Args:
        source_node_type (Optional[str]): Specifies the allowed source node type.
            Defaults to None, allowing all node types.
        target_node_type (Optional[str]): Specifies the allowed target node type.
            Defaults to None, allowing all node types.
        rel_type (Optional[str]): Specifies the allowed relationship type.
            Defaults to None, allowing all relationship types.
        rel_properties (Union[bool, List[str]]): Specifies if relationship properties should
            be included. If a list is provided, only properties with keys in the list
            will be included. If True, all properties are included. Defaults to False.
        llm_type (Optional[str]): The type of the language model. Defaults to None.
            Only openai supports enum param: openai-chat.

    Returns:
        Type[BaseModel]: A relationship model with the specified constraints.

    Raises:
        ValueError: If 'id' is included in the relationship properties list.
    """

    rel_fields: Dict[str, Tuple[Any, Any]] = {
        "source_node_id": (
            str,
            Field(..., description="ID of the source node."),
        ),
        "source_node_type": (
            str,
            optional_enum_field(
                [source_node_type],
                description="The type or label of the source node.",
                input_type="node",
                llm_type=llm_type,
            ),
        ),
        "target_node_id": (
            str,
            Field(..., description="ID of the target node."),
        ),
        "target_node_type": (
            str,
            optional_enum_field(
                [target_node_type],
                description="The type or label of the target node.",
                input_type="node",
                llm_type=llm_type,
            ),
        ),
        "type": (
            str,
            optional_enum_field(
                [rel_type],
                description="The type or label of the relationship.",
                input_type="relationship",
                llm_type=llm_type,
            ),
        ),
    }
    if rel_properties:
        if isinstance(rel_properties, list) and "id" in rel_properties:
            raise ValueError("The relationship property 'id' is reserved and cannot be used.")
        # Map True to empty array
        rel_properties_mapped: List[str] = (
            [] if rel_properties is True else rel_properties
        )

        class Property(BaseModel):
            """A single property consisting of key and value"""

            key: str = optional_enum_field(
                rel_properties_mapped,
                description="Property key.",
                input_type="property",
            )
            value: str = Field(..., description="value")

        rel_fields["properties"] = (
            Optional[List[Property]],
            Field(None, description="List of relationship properties"),
        )
    SimpleRelationship = create_model("SimpleRelationship", **rel_fields)  # type: ignore

    return SimpleRelationship

def create_logical_rel_unstructured(nodes) -> List[Relationship]:
    rels = []
    # find userstory node
    for node in nodes:
        if node[1] == 'Userstory':
            user_story_node = node

    for node in nodes:
        source_node = Node(id=user_story_node[0], type=user_story_node[1])
        target_node = Node(id=node[0], type=node[1])
        if node[1] == 'Persona':
            rels.append(Relationship(
                        source=source_node, target=target_node, type='HAS_PERSONA')
            )
        elif node[1] == 'Action':
            rels.append(Relationship(
                source=source_node, target=target_node, type='HAS_ACTION')
            )
        elif node[1] == 'Entity':
            rels.append(Relationship(
                        source=source_node, target=target_node, type='HAS_ENTITY')
            )
        elif node[1] == 'Benefit':
            rels.append(Relationship(
                        source=source_node, target=target_node, type='HAS_BENEFIT')
            )
    return rels

class UserStoryGraphTransformer:
    """Transform user stories into graph-based documents using a LLM.

    This function is based on LLLMGraphTransformer from langchain_experimental.graph_transformers. 
    It was modified for the specific use case of extracting nodes and relationships from user stories.

    Args:
        llm (BaseLanguageModel): An instance of a language model supporting structured
          output.
        allowed_nodes (List[str], optional): Specifies which node types are
          allowed in the graph. Defaults to Persona, Entity, and Action.
        allowed_relationships (List[str], optional): Specifies which relationship types
          are allowed in the graph. Defaults to TRIGGERS and TARGETS.
        node_properties (Union[bool, List[str]]): If True, the LLM can extract any
          node properties from text. Alternatively, a list of valid properties can
          be provided for the LLM to extract, restricting extraction to those specified.
        relationship_properties (Union[bool, List[str]]): If True, the LLM can extract
          any relationship properties from text. Alternatively, a list of valid
          properties can be provided for the LLM to extract, restricting extraction to
          those specified.
        ignore_tool_usage (bool): Indicates whether the transformer should
          bypass the use of structured output functionality of the language model.
          If set to True, the transformer will not use the language model's native
          function calling capabilities to handle structured output. Defaults to False.

    Example:
        .. code-block:: python
            from langchain_experimental.graph_transformers import UserStoryGraphTransformer
            from langchain_core.documents import Document
            from langchain_openai import ChatOpenAI

            llm=ChatOpenAI(temperature=0)
            transformer = UserStoryGraphTransformer(llm=llm)

            doc = Document(page_content="As a user, I want to sync my data so that I can access my information from anywhere.")
            graph_documents = transformer.convert_to_graph_documents(doc)
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        allowed_nodes: List[str] = ["Persona", "Entity", "Action"],
        allowed_relationships: List[str] = ["TRIGGERS", "TARGETS"],
        node_properties: Union[bool, List[str]] = False,
        relationship_properties: Union[bool, List[str]] = False,
        ignore_tool_usage: bool = False,
    ) -> None:
        self.allowed_nodes = allowed_nodes
        self.allowed_relationships = allowed_relationships
        self._function_call = not ignore_tool_usage

        # Check if the LLM really supports structured output
        if self._function_call:
            try:
                llm.with_structured_output(_Graph)
            except NotImplementedError:
                self._function_call = False
        if not self._function_call:
            if node_properties or relationship_properties:
                raise ValueError(
                    "The 'node_properties' and 'relationship_properties' parameters "
                    "cannot be used in combination with a LLM that doesn't support "
                    "native function calling."
                )
            try:
                import json_repair  # type: ignore

                self.json_repair = json_repair
            except ImportError:
                raise ImportError(
                    "Could not import json_repair python package. "
                    "Please install it with `pip install json-repair`."
                )
            prompt = create_unstructured_prompt(
                allowed_nodes, allowed_relationships
            )
            self.llm = llm
            self.chain = prompt | self.llm

            default_benefit_prompt = benefit_prompt
            self.benefit_chain = default_benefit_prompt | self.llm

        else:
            # Define chain
            try:
                llm_type = llm._llm_type  # type: ignore
            except AttributeError:
                llm_type = None
            schema = create_simple_model(
                allowed_nodes,
                allowed_relationships,
                node_properties,
                llm_type,
                relationship_properties,
            )
            self.structured_llm = llm.with_structured_output(schema, include_raw=True)
            prompt = default_prompt
            self.chain = prompt | self.structured_llm

            default_benefit_prompt = benefit_prompt
            self.benefit_chain = default_benefit_prompt | self.structured_llm

    def process_response(
        self, document: Document, config: Optional[RunnableConfig] = None
    ) -> GraphDocument:
        """
        Processes a single document, transforming it into a graph document using
        an LLM based on the model's schema and constraints.
        """
        text = document.page_content
        raw_schema = self.chain.invoke({"input": text}, config=config)
        benefit_raw_schema = self.benefit_chain.invoke({"input": text})
        
        if self._function_call:
            raw_schema = cast(Dict[Any, Any], raw_schema)
            
            # add user story as a node
            raw_schema['user_story'] = text

            # Extract the nodes
            if benefit_raw_schema: 
                if benefit_raw_schema['parsed'].nodes:
                    nodes = benefit_raw_schema['parsed'].nodes

                    # Filter the nodes to keep only those with type 'Benefit'
                    benefit_nodes = [node for node in nodes if node.type == 'Benefit']

                    raw_schema['parsed'].nodes += benefit_nodes
            
            nodes, relationships = _convert_to_graph_document(raw_schema)
            

        else:
            nodes_set = set()
            relationships = [] 
            
            if not isinstance(benefit_raw_schema, str):
                benefit_raw_schema = benefit_raw_schema.content

            if not isinstance(raw_schema, str):
                raw_schema = raw_schema.content

            # add user story as a node
            nodes_set.add((text, 'Userstory'))

            # checks if the benefit exists
            if len(benefit_raw_schema) > 2:
                # add benefit as a node
                match = re.search(r"Node\(id='(.+)', type='Benefit'\)", benefit_raw_schema)
                if match:
                    benefit_id_part = match.group(1)

                nodes_set.add((benefit_id_part, 'Benefit'))

            parsed_json = self.json_repair.loads(raw_schema)
            if isinstance(parsed_json, dict):
                parsed_json = [parsed_json]
            for rel in parsed_json:
                head_id = normalize_node_id(rel["head"], rel["head_type"])
                tail_id = normalize_node_id(rel["tail"], rel["tail_type"])

                nodes_set.add((head_id, rel["head_type"]))
                nodes_set.add((tail_id, rel["tail_type"]))

                source_node = Node(id=head_id, type=rel["head_type"])
                target_node = Node(id=tail_id, type=rel["tail_type"])
                relationships.append(
                    Relationship(
                        source=source_node, target=target_node, type=rel["relation"]
                    )
                )
            # add logical relationships
            logical_rel = create_logical_rel_unstructured(nodes_set)
            relationships.extend(logical_rel)

            # Create nodes list
            nodes = [Node(id=el[0], type=el[1]) for el in list(nodes_set)]


        return GraphDocument(nodes=nodes, relationships=relationships, source=document)

    def convert_to_graph_documents(
        self, documents: Sequence[Document], config: Optional[RunnableConfig] = None
    ) -> List[GraphDocument]:
        """Convert a sequence of documents into graph documents.

        Args:
            documents (Sequence[Document]): The original documents.
            kwargs: Additional keyword arguments.

        Returns:
            Sequence[GraphDocument]: The transformed documents as graphs.
        """
        return [self.process_response(document, config) for document in documents]