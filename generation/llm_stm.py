
import os
import getpass
import dotenv
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from .model import ChatLLM
dotenv.load_dotenv()


if not os.environ.get("GOOGLE_API_KEY"):
    api_key = getpass.getpass("ENTER GOOGLE_API_KEY: ").strip()
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    else:
        raise RuntimeError(
            "GOOGLE_API_KEY This field cannot be empty."
            "Please provide a valid API key."
        )

LLM = ChatLLM(model="gemini-2.5-flash", provider="gemini", api_key=os.environ.get("GOOGLE_API_KEY")) 

class ChatWithMemory:
    """Manages chatbot workflow with short-term and long-term memory."""

    def __init__(self, model=LLM, summary_threshold=10):
        """
        Args:
            model: Language model instance (must have .invoke method)
            summary_threshold: Number of messages before summarizing
        """
        self.model = model
        self.summary_threshold = summary_threshold

        self.workflow = StateGraph(state_schema=MessagesState)
        self.workflow.add_node("model", self.call_model)
        self.workflow.add_edge(START, "model")

        self.memory = MemorySaver()

    def _build_system_message(self):
        """System prompt configuration."""
        return SystemMessage(content=(
            "Bạn là một nhân viên tư vấn điện thoại nhiệt tình"
            "Hãy trả lời người dùng một cách lịch sự, tên bạn là Lisa, luôn trả lời là Lisa thay vì tôi"
        ))

    def _summarize_messages(self, message_history):
        """Summarize chat history when it exceeds the threshold."""
        summary_prompt = (
            "Distill the above chat messages into a single summary message. "
            "Include as many specific details as you can."
        )
        return self.model.invoke(message_history + [HumanMessage(content=summary_prompt)])

    def call_model(self, state: MessagesState):
        system_message = self._build_system_message()
        message_history = state["messages"][:-1]

        if len(message_history) >= self.summary_threshold:
            last_user_message = state["messages"][-1]
            summary_message = self._summarize_messages(message_history)

            delete_ops = [RemoveMessage(id=m.id) for m in state["messages"]]

            human_message = HumanMessage(content=last_user_message.content)
            response = self.model.invoke([system_message, summary_message, human_message])

            message_updates = [summary_message, human_message, response] + delete_ops
        else:
            message_updates = self.model.invoke([system_message] + state["messages"])

        return {"messages": message_updates}

    def compile(self):
        """Compile workflow with memory checkpoint."""
        return self.workflow.compile(checkpointer=self.memory)
