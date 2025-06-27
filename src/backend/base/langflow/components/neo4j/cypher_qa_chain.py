from langflow.custom import CustomComponent
from langflow.field_typing import Chain, LanguageModel, Neo4jGraph, Prompt
from langflow.io import HandleInput, Output, PromptInput


class GraphCypherQAChainComponent(CustomComponent):
    display_name: str = "Graph Cypher QA Chain"
    description: str = "Chain for question-answering against a Neo4j graph using Cypher."
    icon = "Zap"
    name = "GraphCypherQAChain"

    inputs = [
        HandleInput(name="graph", display_name="Neo4j Graph", input_types=["Neo4jGraph"]),
        HandleInput(name="cypher_llm", display_name="Cypher LLM", input_types=["LanguageModel"]),
        HandleInput(name="qa_llm", display_name="QA LLM", input_types=["LanguageModel"]),
        PromptInput(name="prompt", display_name="Prompt", info="Prompt to use for question answering."),
    ]

    outputs = [
        Output(display_name="Chain", name="chain", field_type=Chain, method="build_chain"),
    ]

    def build_chain(self) -> Chain:
        from langchain.chains import GraphCypherQAChain

        if not isinstance(self.graph, Neo4jGraph):
            raise ValueError("Neo4j Graph must be provided.")
        if not isinstance(self.cypher_llm, LanguageModel):
            raise ValueError("Cypher LLM must be provided.")
        if not isinstance(self.qa_llm, LanguageModel):
            raise ValueError("QA LLM must be provided.")
        if not isinstance(self.prompt, Prompt):
            raise ValueError("Prompt must be provided.")

        return GraphCypherQAChain(
            graph=self.graph,
            cypher_llm=self.cypher_llm,
            qa_llm=self.qa_llm,
            qa_prompt=self.prompt,
            verbose=True,  # Add other parameters as needed
        )
