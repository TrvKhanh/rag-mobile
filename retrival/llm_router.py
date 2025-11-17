from __future__ import annotations
import re
from generation.llm_stm import LLM
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, ValidationError, validator
import os

# Get the directory of the current script
base_dir = os.path.dirname(__file__)

# Load the router prompt
with open(os.path.join(base_dir, '..', 'prompts', 'retrieval', 'router.j2'), 'r', encoding='utf-8') as f:
    ROUTER_PROMPT = f.read()

# Load the strict addition for the router prompt
with open(os.path.join(base_dir, '..', 'prompts', 'retrieval', 'router_strict_addition.txt'), 'r', encoding='utf-8') as f:
    ROUTER_STRICT_ADDITION = f.read()

logger = logging.getLogger("llm_router")
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

model = LLM | StrOutputParser()

# Define simple chat patterns using regex for flexible matching
CHAT_PATTERNS = [
    re.compile(r"^\s*(chào|hi|hello|alo)\b", re.IGNORECASE),
    re.compile(r"\b(cảm ơn|thank you|thanks)\b", re.IGNORECASE),
    re.compile(r"\b(tạm biệt|bye)\b", re.IGNORECASE),
    re.compile(r"\b(bạn là ai|bạn tên gì)\b", re.IGNORECASE),
    re.compile(r"^\s*(oke|ok|tuyệt vời|tốt quá)\s*$", re.IGNORECASE),
]

class ChatOutput(BaseModel):
    router: str
    infor: str

    @validator("router")
    def router_must_be_chat(cls, v):
        if v != "chat":
            raise ValueError("router must be 'chat'")

class RetrievalOutput(BaseModel):
    router: str
    infor: str

    @validator("router")
    def router_must_be_retrieval(cls, v):
        if v != "retrieval":
            raise ValueError("router must be 'retrieval'")
        return v

    @validator("infor")
    def infor_nonempty(cls, v: str):
        if not v or not v.strip():
            raise ValueError("infor must be non-empty")
        return v


class ComparisonOutput(BaseModel):
    router: str
    products: List[str]

    @validator("router")
    def router_must_be_comparison(cls, v):
        if v != "comparison":
            raise ValueError("router must be 'comparison'")
        return v

    @validator("products")
    def products_nonempty(cls, v: List[str]):
        if not v or len(v) < 2:
            raise ValueError("comparison requires >= 2 products")
        cleaned = [p.strip() for p in v if p and p.strip()]
        if len(cleaned) < 2:
            raise ValueError("comparison requires >= 2 non-empty product names")
        return cleaned
    
OUTPUT_SCHEMAS = (ChatOutput, RetrievalOutput, ComparisonOutput)

JSON_OBJ_RE = re.compile(
    r"(\{(?:[^{}]|\{[^{}]*\})*\})", re.DOTALL
)  # greedy-ish extractor for first {...} block


def extract_json_like(text: str) -> Optional[str]:
    """
    Tries to extract a JSON-ish object from the LLM output.
    Returns the first {...} block found, or None.
    """
    if not text:
        return None
    # Remove common markdown fences
    text = re.sub(r"```(?:json|python)?\n?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    # Find first {...} block
    m = JSON_OBJ_RE.search(text)
    if m:
        return m.group(1)
    return None


def strict_json_load(s: str) -> Any:
    """
    Attempt to coerce common Python-literal quirks into valid JSON, then json.loads.
    - Convert single quotes to double quotes when safe.
    - Replace Python True/False/None with true/false/null.
    - Remove trailing commas in objects/arrays.
    This is conservative; prefer that LLM returns valid JSON.
    """
    # Quick sanity: if it's already valid JSON, load immediately
    try:
        return json.loads(s)
    except Exception:
        pass

    # Heuristics to fix common issues:
    t = s.strip()

    # Replace Python booleans/None -> JSON
    t = re.sub(r"\bNone\b", "null", t)
    t = re.sub(r"\bTrue\b", "true", t)
    t = re.sub(r"\bFalse\b", "false", t)

    # Replace single quotes with double quotes when it's likely a JSON-like dict.
    # Only naive replacement where single quotes surround words/phrases.
    # This is inherently heuristic.
    t = re.sub(r"(?<=[:\s,\[])\s*'([^']*)'\s*(?=[,\}\]])", r'"\1"', t)
    t = re.sub(r"(?<=\{)\s*'([^']*)'\s*:", r'"\1":', t)  # keys
    t = re.sub(r":\s*'([^']*)'\s*(?=[,\}])", r': "\1"', t)  # values

    # Remove trailing commas before } or ]
    t = re.sub(r",\s*(\}|])", r"\1", t)

    return json.loads(t)

class Router:
    def __init__(
        self,
        llm_adapter = model,
        max_retries: int = 2,
    ):
        self.llm = llm_adapter
        self.max_retries = max_retries
        # Minimal, strict system prompt encouraging valid JSON only outputs
        self.system_prompt_template = ROUTER_PROMPT

    def _rule_based_fastpath(self, user_input: str) -> Optional[Dict[str, Any]]:
        for p in CHAT_PATTERNS:
            if p.search(user_input):
                logger.debug("Rule-based matched chat pattern.")
                return {"router": "chat", "infor": user_input.strip()}
        return None

    def _build_messages(self, user_input: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt_template},
            {"role": "user", "content": user_input},
        ]

    def _parse_and_validate(self, raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Attempt to extract JSON and validate against Pydantic schemas.
        Returns (parsed_dict, error_message)
        """
        if not raw or not raw.strip():
            return None, "empty response from LLM"

        logger.debug("Raw LLM output: %s", raw)

        json_text = extract_json_like(raw)
        if json_text is None:
            # maybe the model returned a naked JSON without braces or weird formatting
            # as a last resort, try to parse whole output
            json_text = raw.strip()

        try:
            parsed = strict_json_load(json_text)
        except Exception as e:
            logger.debug("JSON load failed: %s", e)
            return None, f"json_load_error: {e}"

        # Validate against schemas
        for schema_cls in OUTPUT_SCHEMAS:
            try:
                obj = schema_cls.parse_obj(parsed)
                # Return canonical dict (pydantic will coerce/clean)
                return obj.dict(), None
            except ValidationError as ve:
                # Not matching this schema; try next
                continue

        return None, "validation failed: does not match any router schema"

    def classify(self, user_input: str) -> Dict[str, Any]:
        """
        Main entrypoint. Returns a validated dict with keys matching the router schemas.
        Guarantees: returns a dict with 'router' field. If parsing fails, default to retrieval fallback.
        """
        # 1) Rule-based fast path
        rule = self._rule_based_fastpath(user_input)
        if rule is not None:
            logger.info("Fast-path rule matched. Returning chat router.")
            return rule

        # 2) LLM classification with retries
        last_error = None
        messages = self._build_messages(user_input)
        for attempt in range(1, self.max_retries + 2):  # initial + max_retries
            logger.debug("Invoking LLM attempt %d for input: %s", attempt, user_input)
            raw = self.llm.invoke(messages)
            parsed, err = self._parse_and_validate(raw)
            if parsed:
                logger.info("LLM returned valid router on attempt %d: %s", attempt, parsed)
                return parsed
            logger.warning("LLM parse/validation failed on attempt %d: %s", attempt, err)
            last_error = err
            # escalate: prepend a stricter instruction
            messages = [
                {"role": "system", "content": self.system_prompt_template + ROUTER_STRICT_ADDITION},
                {"role": "user", "content": user_input},
            ]

        # 3) Fallback: safe default
        logger.error("All attempts failed. Falling back to retrieval. Last error: %s", last_error)
        return {"router": "retrieval", "infor": user_input.strip()}