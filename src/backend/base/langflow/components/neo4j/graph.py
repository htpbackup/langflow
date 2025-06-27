from langflow.custom import CustomComponent
from langflow.field_typing import Neo4jGraph
from langflow.io import Output, SecretStrInput, StrInput


class Neo4jGraphComponent(CustomComponent):
    display_name: str = "Neo4j Graph"
    description: str = "Initializes a Neo4j graph connection."
    icon = "Graph"
    name = "Neo4jGraph"

    inputs = [
        StrInput(name="url", display_name="URL", info="URL for the Neo4j database.", value="bolt://localhost:7687"),
        StrInput(name="username", display_name="Username", info="Username for authentication.", value="neo4j"),
        SecretStrInput(name="password", display_name="Password", info="Password for authentication."),
        StrInput(
            name="database", display_name="Database", info="Name of the Neo4j database.", value="neo4j", advanced=True
        ),
    ]

    outputs = [
        Output(display_name="Graph", name="graph", field_type=Neo4jGraph, method="build_graph"),
    ]

    def build_graph(self) -> Neo4jGraph:
        return Neo4jGraph(
            url=self.url,
            username=self.username,
            password=self.password,
            database=self.database,
        )
