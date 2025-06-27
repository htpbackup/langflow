from langflow.custom import CustomComponent
from langflow.field_typing import BaseChatMessageHistory, Neo4jGraph
from langflow.io import HandleInput, Output, StrInput


class Neo4jChatMessageHistoryComponent(CustomComponent):
    display_name: str = "Neo4j Chat Message History"
    description: str = "Stores chat messages in a Neo4j graph."
    icon = "MessageSquare"
    name = "Neo4jChatMessageHistory"

    inputs = [
        HandleInput(name="graph", display_name="Neo4j Graph", input_types=["Neo4jGraph"]),
        StrInput(name="session_id", display_name="Session ID", info="Identifier for the chat session."),
        StrInput(
            name="node_label",
            display_name="Node Label",
            info="Label for the chat message nodes.",
            value="ChatMessage",
            advanced=True,
        ),
        StrInput(
            name="relationship_type",
            display_name="Relationship Type",
            info="Type of relationship connecting chat messages.",
            value="NEXT_MESSAGE",
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Chat History",
            name="chat_history",
            field_type=BaseChatMessageHistory,
            method="build_chat_history",
        ),
    ]

    def build_chat_history(self) -> BaseChatMessageHistory:
        from langchain_neo4j.chat_message_histories import Neo4jChatMessageHistory

        if not isinstance(self.graph, Neo4jGraph):
            raise ValueError("Neo4j Graph must be provided.")

        return Neo4jChatMessageHistory(
            graph=self.graph,
            session_id=self.session_id,
            node_label=self.node_label,
            relationship_type=self.relationship_type,
        )
