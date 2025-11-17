
import os
import getpass
import dotenv
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from .model import ChatLLM
dotenv.load_dotenv()

# Get the directory of the current script
base_dir = os.path.dirname(__file__)

# Load the summary prompt
with open(os.path.join(base_dir, '..', 'prompts', 'generation', 'summary.j2'), 'r', encoding='utf-8') as f:
    SUMMARY_PROMPT = f.read()


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

    def _summarize_messages(self, message_history):
        """Summarize chat history when it exceeds the threshold."""
        summary_prompt = SUMMARY_PROMPT
        return self.model.invoke(message_history + [HumanMessage(content=summary_prompt)])

    def call_model(self, state: MessagesState):
        message_history = state["messages"][:-1]

        if len(message_history) >= self.summary_threshold:
            last_user_message = state["messages"][-1]
            summary_message = self._summarize_messages(message_history)

            delete_ops = [RemoveMessage(id=m.id) for m in state["messages"]]

            human_message = HumanMessage(content=last_user_message.content)
            response = self.model.invoke([summary_message, human_message])

            message_updates = [summary_message, human_message, response] + delete_ops
        else:
            message_updates = self.model.invoke(state["messages"])

        return {"messages": message_updates}

    def compile(self):
        """Compile workflow with memory checkpoint."""
        return self.workflow.compile(checkpointer=self.memory)
