from langflow.custom import CustomComponent
from langflow.field_typing import Embeddings, VectorStore
from langflow.io import HandleInput, Output, SecretStrInput, StrInput


class Neo4jVectorComponent(CustomComponent):
    display_name: str = "Neo4j Vector Store"
    description: str = "Creates a Neo4j vector store for similarity search."
    icon = "Database"
    name = "Neo4jVector"

    inputs = [
        StrInput(name="url", display_name="URL", info="URL for the Neo4j database.", value="bolt://localhost:7687"),
        StrInput(name="username", display_name="Username", info="Username for authentication.", value="neo4j"),
        SecretStrInput(name="password", display_name="Password", info="Password for authentication."),
        HandleInput(name="embedding", display_name="Embedding", input_types=["Embeddings"]),
        StrInput(name="index_name", display_name="Index Name", info="Name of the vector index.", value="vector"),
        StrInput(
            name="embedding_node_property",
            display_name="Embedding Node Property",
            info="Property name for storing embeddings on nodes.",
            value="embedding",
            advanced=True,
        ),
        StrInput(
            name="text_node_property",
            display_name="Text Node Property",
            info="Property name for storing text on nodes.",
            value="text",
            advanced=True,
        ),
        StrInput(
            name="retrieval_query",
            display_name="Retrieval Query",
            info="Cypher query for retrieving data during search.",
            value="RETURN node.text AS text, score, node {.*} AS metadata",
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Vector Store", name="vector_store", field_type=VectorStore, method="build_vector_store"),
    ]

    def build_vector_store(self) -> VectorStore:
        from langchain_neo4j.vectorstores import Neo4jVector

        if not isinstance(self.embedding, Embeddings):
            raise ValueError("Embedding must be provided.")

        return Neo4jVector.from_existing_index(
            embedding=self.embedding,
            url=self.url,
            username=self.username,
            password=self.password,
            index_name=self.index_name,
            embedding_node_property=self.embedding_node_property,
            text_node_property=self.text_node_property,
            retrieval_query=self.retrieval_query,
        )
