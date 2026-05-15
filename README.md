# Relation-Enriched User-Story Graphs

This repository contains the implementation, prompts, evaluation material, and supplementary artifacts developed for a master's thesis on extending user-story knowledge graphs with explicit semantic relations.

The work builds on the UserStoryGraphTransformer (USGT) and adds two semantic relation layers:

- **Intra-story relations** between actions within a single user story:
  - `ENABLES`
  - `BLOCKS`
  - `NONE`

- **Inter-story relations** between user stories in a backlog:
  - `DUPLICATE`
  - `CONFLICT`
  - `DEPENDS_ON`
  - `NONE`

The resulting relations are represented as additional edges in a graph-based user-story representation and explored through Neo4j queries.

---

## Repository Contents

The repository includes:

- source code for relation candidate construction and graph enrichment,
- prompt templates for relation classification,
- prompt templates for targeted synthetic transformations,
- evaluation scripts and result-processing utilities,
- curated evaluation material used in the thesis,
- Neo4j Cypher queries used in the applicability study,
- supplementary files supporting reproducibility of the reported analyses.

---

## Project Scope

The repository supports the following parts of the thesis workflow:

1. **Structural graph foundation**
   - Use of an existing USGT-based user-story graph representation.

2. **Intra-story relation extraction**
   - Construction of action pairs within individual stories.
   - Classification of action-to-action semantic relations.

3. **Inter-story relation extraction**
   - Retrieval-based candidate construction using embedding similarity and Okapi BM25.
   - Classification of semantic relations between user-story pairs.

4. **Synthetic transformation pipeline**
   - Controlled generation of additional examples for underrepresented relation types.
   - Manual curation of generated material before evaluation use.

5. **Evaluation and analysis**
   - Quantitative evaluation of relation classification.
   - Qualitative error analysis.
   - Graph-based applicability examples in Neo4j.

---

## Data and Materials

The repository contains processed evaluation material, curated synthetic examples, prompt files, result files, and supplementary artifacts required to reproduce the main analyses reported in the thesis, where redistribution is permitted.

The work builds on external user-story resources, including:

- the requirements dataset collection published by Fabiano Dalpiaz, including the `g14-datahub` backlog used in this study;
- the qualified user-story resources released by Sathurshan Arulmohan, Sébastien Mosser, and Marie-Jean Meurs through the ACE Design project and associated dataset release.

These external resources are not original contributions of this repository. This repository contributes the semantic-relation extraction pipeline, curated evaluation materials derived for the thesis, prompt files, result-processing code, and Neo4j applicability queries.

Where original source material is not redistributed directly, the repository provides derived files where permitted, preprocessing scripts, or documentation explaining how the materials were used.

---

## Data Sources and Attribution

### User-Story Backlog Data

The user-story backlog material used in this thesis is based on the requirements datasets published by Fabiano Dalpiaz. The study uses the `g14-datahub` backlog from this collection.

- Fabiano Dalpiaz
  [https://zenodo.org/records/13880060](https://zenodo.org/records/13880060)

### Qualified User-Story Annotations

For action-level information used in parts of the intra-story evaluation setup, this work refers to the qualified user-story resources released by Sathurshan Arulmohan, Sébastien Mosser, and Marie-Jean Meurs.

- ACE Design
  [https://github.com/ace-design/qualified-user-stories](https://github.com/ace-design/qualified-user-stories)

### UserStoryGraphTransformer

This thesis builds on the UserStoryGraphTransformer (USGT) as the structural graph-construction foundation. Add the archived software or dataset record here once the exact bibliographic metadata of the intended source has been confirmed.

- USGT archive  
  [https://zenodo.org/records/14254059](https://zenodo.org/records/14254059)

---

## Directory Structure

```text
.
├── data/
│   ├── processed/
│   ├── evaluation/
│   └── synthetic/
│
├── prompts/
│   ├── classification/
│   └── transformations/
│
├── src/
│   ├── intra_story/
│   ├── inter_story/
│   ├── retrieval/
│   ├── graph/
│   └── utils/
│
├── evaluation/
│   ├── metrics/
│   ├── outputs/
│   └── error_analysis/
│
├── neo4j/
│   └── applicability_queries.cypher
│
├── results/
├── README.md
├── requirements.txt
└── LICENSE
```

## Citation

If you use this repository, please cite the associated thesis.

@mastersthesis{yourkey,
  author = {Anna Tatinyan},
  title  = {Extending User Story Knowledge Graphs with Intra- and Inter-Story Relation Using Large Language Models},
  school = {Humboldt-Universität zu Berlin},
  year   = {2026}
}
