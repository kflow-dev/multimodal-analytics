"""Shared runtime helpers for local AIFluent execution."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_LOCAL_LIGHTRAG_DIR = (
    "./data/vendor/000_LightRAG"
)


def bootstrap_lightrag_source() -> None:
    """Allow imports from a local LightRAG checkout before importing aifluent."""
    source_dir = os.getenv("LIGHTRAG_SOURCE_DIR", DEFAULT_LOCAL_LIGHTRAG_DIR)
    root = Path(source_dir).expanduser().resolve()
    package_init = root / "lightrag" / "__init__.py"

    if not package_init.exists():
        return

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


bootstrap_lightrag_source()

from aifluent import AIFluent, AIFluentConfig


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def import_lightrag_openai():
    try:
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed
        from lightrag.utils import wrap_embedding_func_with_attrs
    except ImportError as exc:
        raise RuntimeError(
            "LightRAG OpenAI helpers are unavailable. Install project dependencies first."
        ) from exc

    return openai_complete_if_cache, openai_embed, wrap_embedding_func_with_attrs


def build_model_adapters():
    api_key = require_env("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    llm_model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
    vision_model = os.getenv("OPENAI_VISION_MODEL", llm_model)
    embedding_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
    embedding_dim = int(os.getenv("OPENAI_EMBED_DIM", "3072"))

    openai_complete_if_cache, openai_embed, wrap_embedding_func_with_attrs = (
        import_lightrag_openai()
    )

    async def llm_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> str:
        return await openai_complete_if_cache(
            llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    @wrap_embedding_func_with_attrs(
        embedding_dim=embedding_dim,
        max_token_size=8192,
    )
    async def embedding_func(texts: list[str]) -> np.ndarray:
        return await openai_embed.func(
            texts,
            model=embedding_model,
            api_key=api_key,
            base_url=base_url,
        )

    async def vision_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        image_data: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> str:
        if messages is not None:
            return await openai_complete_if_cache(
                vision_model,
                "",
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )

        if image_data:
            multimodal_messages = [
                {
                    "role": "system",
                    "content": system_prompt or "You are a helpful multimodal assistant.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                        },
                    ],
                },
            ]
            return await openai_complete_if_cache(
                vision_model,
                "",
                messages=multimodal_messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )

        return await openai_complete_if_cache(
            vision_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    return llm_model_func, embedding_func, vision_model_func


def build_rag() -> AIFluent:
    llm_model_func, embedding_func, vision_model_func = build_model_adapters()
    config = AIFluentConfig(
        parser=os.getenv("PARSER", "mineru"),
        parse_method=os.getenv("PARSE_METHOD", "auto"),
        working_dir=os.getenv("WORKING_DIR", "./rag_storage"),
        parser_output_dir=os.getenv("OUTPUT_DIR", "./output"),
    )

    Path(config.working_dir).mkdir(parents=True, exist_ok=True)
    Path(config.parser_output_dir).mkdir(parents=True, exist_ok=True)

    return AIFluent(
        config=config,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        vision_model_func=vision_model_func,
    )
