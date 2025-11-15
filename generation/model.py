import os
import time
from typing import Iterator, Literal, Any, Optional, List
import logging
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.messages import  (
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
    """Initialize the LLM client"""
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
        self.enable_concurrency = bool(self.ollama_config.get("ENABLE_CONCURRENCY", False))
        self.model = model_name

        self._init_llm()
    def _init_llm(self):
        if self.provider == "ollama":
            import ollama

            host = self.ollama_config.get("OLLAMA_HOST", "http://localhost:11434")
            timeout = self.ollama_config.get("OLLAMA_REQUEST_TIMEOUT", 240.0)

            self.client = ollama.Client(host=host, timeout=timeout)
            try:
                self.client.show(self.model_name)
            except ollama.ResponseError as e:
                if e.status_code == 404:
                    _logger.warning(f"Model {self.model_name} not found. Pulling...")
                    for progress in self.client.pull(self.model_name, stream=True):
                        _logger.info("Pull model")
            except Exception as e:
                raise LLMError(
                    f"Unable to download {self.model_name} from Ollama: {str(e)}"
                )
            try:
                os.environ["OLLAMA_KEEP_ALIVE"] = str(
                    self.ollama_config.get("OLLAMA_KEEP_ALIVE", -1)
                )
                if self.enable_concurrency:
                    if self.device == "cuda":
                        os.environ["OLLAMA_NUM_GPU"] = str(
                            self.ollama_config.get(
                                "OLLAMA_NUM_GPU", self.num_workers // 2
                            )
                        )
                        os.environ["OLLAMA_NUM_PARALLEL"] = str(
                            self.ollama_config.get(
                                "OLLAMA_NUM_PARALLEL", self.num_workers * 8
                            )
                        )
                        os.environ["OLLAMA_GPU_LAYERS"] = str(
                            self.ollama_config.get("OLLAMA_GPU_LAYERS", "all")
                        )
                    elif self.device == "mps":
                        os.environ["OLLAMA_NUM_GPU"] = str(
                            self.ollama_config.get("OLLAMA_NUM_GPU", 1)
                        )
                        os.environ["OLLAMA_NUM_THREAD"] = str(
                            self.ollama_config.get(
                                "OLLAMA_NUM_THREAD", self.num_workers
                            )
                        )
                        os.environ["OLLAMA_NUM_PARALLEL"] = str(
                            self.ollama_config.get(
                                "OLLAMA_NUM_PARALLEL", self.num_workers * 8
                            )
                        )
                    else:
                        os.environ["OLLAMA_NUM_THREAD"] = str(
                            self.ollama_config.get(
                                "OLLAMA_NUM_THREAD", self.num_workers
                            )
                        )
                        os.environ["OLLAMA_NUM_PARALLEL"] = str(
                            self.ollama_config.get(
                                "OLLAMA_NUM_PARALLEL", self.num_workers * 10
                            )
                        )
            except Exception as e:
                raise LLMError(f"Unable to initialize Ollama client: {str(e)}")
        elif self.provider == "gemini":
            try:
                from google import genai
            except ImportError:
                raise ImportError(
                    "Gemini is not installed. Please install it using pip install pip install -U google-genai."
                )

            try:
                self._genai = genai
                self.client = self._genai.Client(api_key=self.api_key)
            except Exception as e:
                raise LLMError(f"Unable to initialize Gemini client: {str(e)}")
    
    def _response(self, input: str) -> Any:
        response = ""
        max_retries = 3
        base_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                if self.provider == "ollama":
                    if self.stream:
                        response = self.client.generate(
                            model= self.model_name,
                            prompt= input,
                            stream=True
                        )
                    else:
                        response = self.client.generate(
                            model= self.model_name,
                            prompt= input,
                        ).response
                elif self.provider == "gemini":
                    if self.stream:
                        response = self.client.models.generate_content_stream(
                            model=self.model_name,
                            contents= input
                        )
                    else:
                        response = self.client.models.generate_content(
                            model= self.model_name,
                            contents= input,
                        ).text
                else:
                    raise ValueError(f"Unknown provider: {self.provider}")
                
                
                return response
                
            except Exception as e:
                is_503_error = False
                error_str = str(e)
                
                if genai_errors and isinstance(e, genai_errors.ServerError):
                    if hasattr(e, 'status_code') and getattr(e, 'status_code', None) == 503:
                        is_503_error = True
                    elif "503" in error_str or "UNAVAILABLE" in error_str:
                        is_503_error = True
              
                elif "503" in error_str or "overloaded" in error_str.lower() or "UNAVAILABLE" in error_str:
                    is_503_error = True
                
                if is_503_error and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) 
                    _logger.warning(
                        f"Gemini API overloaded (503). Retrying in {delay}s... "
                        f"(Attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                    continue
                else:
                
                    if is_503_error:
                        _logger.error(
                            f"Gemini API still overloaded after {max_retries} attempts. "
                            "Please try again later."
                        )
                    _logger.exception("Generation failed")
                    raise RuntimeError(f"Generation failed for {self.provider}: {e}") from e

        raise RuntimeError(f"Generation failed for {self.provider} after {max_retries} attempts")

class ChatLLM(BaseChatModel):

    provider: Literal["ollama", "gemini"]
    model_name: str =  Field(alias="model")
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
        prompt = "\n".join([text.content for text in messages])
        output = self._llm._response(prompt)
        text_generate = ChatGeneration(message=AIMessage(content=output))

        return ChatResult(generations=[text_generate])
    
    def _stream(
        self,
        messages: List[BaseMessage], 
        stop: List[str] | None = None, 
        run_manager: Optional[CallbackManagerForLLMRun] = None, 
        stream: bool = True,
        **kwargs
        ) ->  Iterator[ChatGenerationChunk]:
       
        prompt = "\n".join([text.content for text in messages])
        response_stream = self._llm._response(prompt)
        self._llm.stream = True
        
        if self.provider == "ollama":
            for chunk in response_stream:
                yield ChatGenerationChunk(message=AIMessageChunk(content=chunk))

        elif self.provider == "gemini":
            for chunk in response_stream:
                yield ChatGenerationChunk(message=AIMessageChunk(content=chunk))

    @property
    def _llm_type(self) -> str:
        return f"custom-{self._llm.provider}"
    
