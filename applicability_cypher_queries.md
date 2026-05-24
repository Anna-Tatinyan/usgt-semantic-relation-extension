# Cypher Queries Used in the Applicability Study

This file contains the Cypher queries used to generate the graph views and focused applicability examples discussed in the thesis applicability study.

## 1. Structural Baseline View

Retrieves user stories together with their `Persona` and `Action` nodes. Used to generate the structural baseline material from which the persona-centered views were selected.

```cypher
MATCH (s:UserStory)
OPTIONAL MATCH (s)-[rp]-(p:Persona)
OPTIONAL MATCH (s)-[ra]-(a:Action)
RETURN DISTINCT s, rp, p, ra, a;
```

## 2. Full Relation-Enriched View

Extends the structural baseline by adding inter-story and intra-story semantic relations. Used to generate the full enriched graph.

```cypher
MATCH (s:UserStory)
OPTIONAL MATCH (s)-[rp]-(p:Persona)
OPTIONAL MATCH (s)-[ra]-(a:Action)
OPTIONAL MATCH (s)-[rs:DUPLICATE|DEPENDS_ON|CONFLICT]-(t:UserStory)
OPTIONAL MATCH (a)-[ri:ENABLES|BLOCKS]-(b:Action)
RETURN DISTINCT s, rp, p, ra, a, rs, t, ri, b;
```

## 3. Stored Semantic Relation Counts

Counts the stored semantic edges in the enriched graph.

```cypher
MATCH ()-[r]->()
WHERE type(r) IN [
  'DUPLICATE',
  'DEPENDS_ON',
  'CONFLICT',
  'ENABLES',
  'BLOCKS'
]
RETURN type(r) AS semantic_relation, count(r) AS relation_count
ORDER BY semantic_relation;
```

## 4. Inter-Story Semantic Relations

Retrieves the inter-story relations between user stories.

```cypher
MATCH (a:UserStory)-[r:DUPLICATE|CONFLICT|DEPENDS_ON]-(b:UserStory)
RETURN DISTINCT a, r, b;
```

## 5. Conflict Relations with Dependency Context

Retrieves conflict relations and the neighboring dependency relations of both involved user stories.

```cypher
MATCH (a:UserStory)-[conf:CONFLICT]-(b:UserStory)
OPTIONAL MATCH (a)-[depA:DEPENDS_ON]-(x:UserStory)
OPTIONAL MATCH (b)-[depB:DEPENDS_ON]-(y:UserStory)
RETURN a, conf, b, depA, x, depB, y;
```

## 6. Duplicate and Conflict Relations

Retrieves `DUPLICATE` and `CONFLICT` relations between user stories.

```cypher
MATCH (a:UserStory)-[r:DUPLICATE|CONFLICT]-(b:UserStory)
RETURN DISTINCT a, r, b;
```

## 7. Dependency Relations

Retrieves direct `DEPENDS_ON` edges between pairs of user stories. When several edges share intermediate stories, they appear in the graph view as dependency-chain structures.

```cypher
MATCH (a:UserStory)-[r:DEPENDS_ON]->(b:UserStory)
RETURN DISTINCT a, r, b;
```

## 8. Focused Intra-Story Relations with User-Story Context

Retrieves a focused subset of user stories together with their `Action` nodes, structural `HAS_ACTION` relations, and intra-story `ENABLES` or `BLOCKS` relations between those actions. The subset is used because displaying all relevant user stories, structural links, and intra-story semantic relations in a single graph view would reduce label legibility.

```cypher
MATCH (s:UserStory)-[:HAS_ACTION]->(:Action)
      -[:ENABLES|BLOCKS]->(:Action)
      <-[:HAS_ACTION]-(s)
WITH DISTINCT s
LIMIT 4
MATCH (s)-[hs1:HAS_ACTION]->(a1:Action)
      -[r:ENABLES|BLOCKS]->(a2:Action)
      <-[hs2:HAS_ACTION]-(s)
RETURN DISTINCT s, hs1, a1, r, a2, hs2;
```
