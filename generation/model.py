import os
import time
from typing import Iterator, Literal, Any, Optional, List
import logging
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    AIMessageChunk,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from pydantic import Field, PrivateAttr

_logger = logging.getLogger(__name__)

try:
    from google.genai import errors as genai_errors
except ImportError:
    genai_errors = None


class LLMError(Exception):
    pass


class LLM:
    def __init__(
        self,
        provider: Literal["ollama", "gemini"],
        model_name: str,
        api_key: str = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        ollama_config: dict | None = None,
        stream: bool = False,
        device: Literal["cuda", "mps", None] = None,
        num_workers: int = 1,
        **kwargs: Any,
    ):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.ollama_config = ollama_config or {}
        self.stream = stream
        self.kwargs = kwargs
        self.device = device
        self.num_workers = num_workers
        self.model = model_name

        self._init_llm()


    def _init_llm(self):
        if self.provider == "ollama":
            import ollama

            host = self.ollama_config.get("OLLAMA_HOST", "http://localhost:11434")
            timeout = self.ollama_config.get("OLLAMA_REQUEST_TIMEOUT", 240.0)

            self.client = ollama.Client(host=host, timeout=timeout)

        elif self.provider == "gemini":
            from google import genai
            self._genai = genai
            self.client = genai.Client(api_key=self.api_key)


    def _response(self, input: str) -> Any:

        if self.provider == "ollama":
            if self.stream:
                return self.client.generate(
                    model=self.model_name, prompt=input, stream=True
                )
            else:
                res = self.client.generate(model=self.model_name, prompt=input)
                return res["response"]  

        if self.provider == "gemini":
            if self.stream:
                return self._gemini_stream(input)
            else:
                return self._gemini_no_stream(input)

        raise ValueError(f"Unknown provider {self.provider}")


    def _gemini_no_stream(self, input: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=input
        )
        return response.text or ""


    def _gemini_stream(self, input: str):
        for chunk in self.client.models.generate_content_stream(
            model=self.model_name,
            contents=input
        ):
            
            if hasattr(chunk, "text") and chunk.text:
                yield chunk.text


class ChatLLM(BaseChatModel):

    provider: Literal["ollama", "gemini"]
    model_name: str = Field(alias="model")
    api_key: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 0.9
    device: Literal["cuda", "mps", None] = None
    num_workers: int = 1

    _llm: LLM = PrivateAttr()

    def model_post_init(self, __context):
        self._llm = LLM(
            provider=self.provider,
            model_name=self.model_name,
            api_key=self.api_key,
            temperature=self.temperature,
            top_p=self.top_p,
            device=self.device,
            num_workers=self.num_workers,
        )


    def _generate(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> ChatResult:

        prompt = "\n".join([msg.content for msg in messages])
        self._llm.stream = False

        output = self._llm._response(prompt)  

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=output))]
        )


    def _stream(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> Iterator[ChatGenerationChunk]:

        prompt = "\n".join([msg.content for msg in messages])
        self._llm.stream = True

        stream = self._llm._response(prompt)

        for chunk in stream:
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=chunk)
            )

    @property
    def _llm_type(self) -> str:
        return f"custom-{self._llm.provider}"
