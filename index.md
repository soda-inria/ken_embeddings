
# Relational Data Embeddings for Feature Enrichment with Background Information

For many machine-learning tasks, **augmenting the data at hand with features built from external sources is key to improving performance**. For instance, estimating housing prices benefits from background information on the location, such as the population density or the average income. However, this information must often be assembled across many tables, requiring time and expertise from the data scientist.

Instead, we propose to replace human-crafted features by **vectorial representations of entities** (*e.g.* cities) that capture the corresponding information. We represent the relational data on the entities as a **graph** and adapt
graph-embedding methods to create feature vectors for each entity. We show that two technical ingredients are crucial: modeling well the different **relationships** between entities, and capturing **numerical** attributes. For this, we leverage **knowledge graph embedding** methods. Although they were primarily designed for graph completion purposes, we show that they can serve as powerful feature extractors. However, they only model discrete entities, while creating good feature vectors from relational data also requires capturing numerical attributes. We thus introduce **KEN** (**K**nowledge **E**mbedding with **N**umbers), a module that extends knowledge graph embedding models to numerical values.

![embedding_pipeline](/assets/figures/embedding_pipeline.png)
*Our proposed embedding pipeline for feature enrichment.*

We thoroughly evaluate approaches to enrich features with background information on 7 prediction tasks. We show that a good embedding model coupled with KEN can perform better than manually handcrafted features, while requiring **much less human effort**. It is also competitive with combinatorial feature engineering methods, but is much more **scalable**. Our approach can be applied to huge databases, creating **general-purpose feature vectors reusable in various downstream tasks**.

# Augmenting your data with information from Wikipedia

We describe in this section how to augment your data with information from Wikipedia. For this, we leverage [YAGO3](https://yago-knowledge.org/downloads/yago-3), a large knowledge base derived from Wikipedia, and apply our embedding pipeline to generate vectors for many entities. These pretrained embeddings are readily available in parquet files that you can download [here](#downloading-entity-embeddings).

### Description of YAGO3
YAGO3 is a large

### Embedding method

### Visualizing entity embeddings

![entity_density](/assets/figures/entity_density.png)
*2D visualization of entity embeddings learned from YAGO3 (brighter zones correspond to higher entity densities).*

![entity_types](/assets/figures/entity_types.png)
*2D visualization of entity embeddings learned from YAGO3, colored by their types.*

### Downloading entity embeddings

some text and [here is possible to download the file in PDF][1]

[1]:{{ site.url }}/assets/data/emb_person.parquet

# Publications

