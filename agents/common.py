# agents/common.py
# This file contains common objects and functions that can be shared across multiple agents or modules.
# Just for reducing redundacy
import os
import chainlit as cl
from langchain.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

# from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.chat_models import init_chat_model

# Load environment variables once
load_dotenv()

# Shared LLM instance
api_key = os.getenv("OPENAI_API_KEY")
claude_api_key = os.getenv("CLAUDE_API_KEY")
# Model settings. The defaults are the models this project was developed with; set the
# variables in .env to use others (see README).
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_CODE_MODEL = os.getenv("LLM_CODE_MODEL", "gpt-4o")
# A whole program comes back in one structured reply, so the cap must fit a long program;
# 4096 tokens can cut it off (current gpt-4o allows 16384).
LLM_CODE_MAX_TOKENS = int(os.getenv("LLM_CODE_MAX_TOKENS", "16384"))
# Reasoning models accept only their default temperature and reject 0. An empty
# LLM_CODE_TEMPERATURE leaves the temperature out of the requests.
_code_temperature = os.getenv("LLM_CODE_TEMPERATURE", "0").strip()

# Initialize OpenAI LLMs. stream_usage: streamed replies report their token usage too.
llm = init_chat_model(api_key=api_key, model=LLM_MODEL, model_provider="openai", stream_usage=True)
llm_code = init_chat_model(
    api_key=api_key,
    model=LLM_CODE_MODEL,
    model_provider="openai",
    temperature=float(_code_temperature) if _code_temperature else None,
    max_tokens=LLM_CODE_MAX_TOKENS,
    stream_usage=True,
)

# Initialize Claude LLM
""" llm_code_claude = ChatAnthropic(
    api_key=claude_api_key,
    model="claude-3-7-sonnet-latest",
    temperature=0,
    max_tokens=4096,
    disable_streaming=True,
    thinking={"type":"disabled"},
) """
# No agent calls this model at the moment. It is created only when a key is set, because the
# Anthropic client refuses to start without one, which stopped the whole app.
llm_code_claude = (
    init_chat_model(
        api_key=claude_api_key,
        model="claude-3-5-sonnet-latest",
        model_provider="anthropic",
        temperature=0,
        max_tokens=4096,
    )
    if claude_api_key
    else None
)

# Export common objects or functions
__all__ = ["cl", "PydanticOutputParser", "llm", "llm_code", "llm_code_claude"]
