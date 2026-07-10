"""Recording and replay wrappers for Graphiti's LLM and embedder clients.

THE RECORDING CONTRACT (mirrored by nacre-core's `model` module — keep in sync):

An LLM exchange is keyed by the request as the *pipeline* built it — the
messages exactly as the prompt function returned them, BEFORE
`LLMClient.generate_response` mutates them (it appends the response-model
JSON schema to the last message and a language instruction to the system
message). The Rust pipeline reproduces the prompt-function output, so that
is the replayable identity; the mutations are provider-plumbing that each
real client applies internally.

Key shape (matches nacre-core `CompletionRequest` serialization):
    {"messages": [{"role": ..., "content": ...}],
     "schema_name": response_model.__name__ (or "" when None),
     "max_tokens": <int, omitted when the caller passed None>,
     "model_size": "small" | "medium"}

Embedding exchanges are keyed as
    {"inputs": [...], "model_id": ...}
with the response being one vector per input, in order (matches
nacre-core `ReplayEmbedder::request_value`).

Recordings are JSON arrays of {"request": ..., "response": ...}, matched by
canonical JSON (sorted keys) — the same format the Rust `RecordingStore`
loads. Replay clients fail loudly on a miss; they never call out.
"""

from __future__ import annotations

import json
from typing import Any

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.prompts.models import Message
from pydantic import BaseModel


def canonical_key(request: dict) -> str:
    """Canonical JSON — matches nacre-core's `canonical_json`."""
    return json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


class RecordingLog:
    """An append-only list of exchanges, saved as a recordings JSON file."""

    def __init__(self) -> None:
        self.recordings: list[dict] = []

    def record(self, request: dict, response: Any) -> None:
        self.recordings.append({'request': request, 'response': response})

    def save(self, path) -> None:
        path.write_text(
            json.dumps(self.recordings, ensure_ascii=False, indent=1, sort_keys=True) + '\n'
        )

    @staticmethod
    def load_index(path) -> dict[str, Any]:
        recordings = json.loads(path.read_text())
        return {canonical_key(r['request']): r['response'] for r in recordings}


def llm_request_key(
    messages: list[Message],
    response_model: type[BaseModel] | None,
    max_tokens: int | None,
    model_size: ModelSize,
) -> dict:
    request: dict = {
        'messages': [{'role': m.role, 'content': m.content} for m in messages],
        'schema_name': response_model.__name__ if response_model is not None else '',
        'model_size': model_size.value,
    }
    if max_tokens is not None:
        request['max_tokens'] = max_tokens
    return request


class RecordingLLMClient(LLMClient):
    """Wraps a real LLMClient; records every generate_response exchange."""

    def __init__(self, inner: LLMClient, log: RecordingLog) -> None:
        super().__init__(config=None, cache=False)
        self.inner = inner
        self.log = log

    async def _generate_response(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError('RecordingLLMClient delegates generate_response entirely')

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
        *,
        attribute_extraction: bool = False,
    ) -> dict[str, Any]:
        # Key by the PRE-mutation messages; hand the inner client copies,
        # since generate_response mutates message contents in place.
        request = llm_request_key(messages, response_model, max_tokens, model_size)
        copies = [m.model_copy(deep=True) for m in messages]
        response = await self.inner.generate_response(
            copies,
            response_model=response_model,
            max_tokens=max_tokens,
            model_size=model_size,
            group_id=group_id,
            prompt_name=prompt_name,
            attribute_extraction=attribute_extraction,
        )
        self.log.record(request, response)
        return response


class ReplayLLMClient(LLMClient):
    """Serves recorded generate_response results; fails loudly on a miss."""

    def __init__(self, index: dict[str, Any]) -> None:
        super().__init__(config=None, cache=False)
        self.index = index

    async def _generate_response(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError('ReplayLLMClient serves recordings only')

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
        *,
        attribute_extraction: bool = False,
    ) -> dict[str, Any]:
        request = llm_request_key(messages, response_model, max_tokens, model_size)
        key = canonical_key(request)
        if key not in self.index:
            raise RuntimeError(
                f'no recording matches this request (replay refuses to guess): {key}'
            )
        return self.index[key]


def _embedder_model_id(inner: EmbedderClient) -> str:
    config = getattr(inner, 'config', None)
    return getattr(config, 'embedding_model', None) or type(inner).__name__


class RecordingEmbedder(EmbedderClient):
    """Wraps a real EmbedderClient; records every embedding exchange."""

    def __init__(self, inner: EmbedderClient, log: RecordingLog) -> None:
        self.inner = inner
        self.log = log
        self.model_id = _embedder_model_id(inner)

    async def create(self, input_data) -> list[float]:
        if not isinstance(input_data, str):
            raise TypeError(f'only str inputs are recorded, got {type(input_data)!r}')
        vector = await self.inner.create(input_data)
        self.log.record({'inputs': [input_data], 'model_id': self.model_id}, [vector])
        return vector

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        vectors = await self.inner.create_batch(input_data_list)
        self.log.record({'inputs': input_data_list, 'model_id': self.model_id}, vectors)
        return vectors


class ReplayEmbedderClient(EmbedderClient):
    """Serves recorded embeddings; fails loudly on a miss."""

    def __init__(self, index: dict[str, Any], model_id: str) -> None:
        self.index = index
        self.model_id = model_id

    def _lookup(self, inputs: list[str]) -> list[list[float]]:
        key = canonical_key({'inputs': inputs, 'model_id': self.model_id})
        if key not in self.index:
            raise RuntimeError(f'no embedding recording for: {key}')
        return self.index[key]

    async def create(self, input_data) -> list[float]:
        if not isinstance(input_data, str):
            raise TypeError(f'only str inputs are replayable, got {type(input_data)!r}')
        return self._lookup([input_data])[0]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        return self._lookup(input_data_list)


class FailLoudCrossEncoder(CrossEncoderClient):
    """Capture uses RRF-only search configs; any cross-encoder use is a bug
    in the capture setup, not something to paper over."""

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        raise RuntimeError(
            'cross-encoder invoked during capture — the golden traces are '
            'defined over RRF search only (see AGENTS.md "not ported")'
        )
