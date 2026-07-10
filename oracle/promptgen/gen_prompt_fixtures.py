#!/usr/bin/env python3
"""Render graphiti's prompt functions to committed fixtures.

Loads prompt modules from the pinned reference clone (../refs/graphiti,
v0.29.2) WITHOUT installing graphiti or its dependencies: `pydantic` is
stubbed and the package __init__ files are bypassed, so stock python3 is
enough. For each prompt function, a canned context is rendered and the
resulting messages are written to
crates/nacre-core/tests/fixtures/prompts/<module>.json.

The Rust port replays the same contexts and must reproduce every message
byte-for-byte (tests/prompt_fidelity.rs). Regenerating fixtures is a
manual, deliberate act — run this script only when re-pinning Graphiti.

Usage: python3 oracle/promptgen/gen_prompt_fixtures.py
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import json
import sys
import types
from pathlib import Path

NACRE = Path(__file__).resolve().parents[2]
GRAPHITI = NACRE.parent / 'refs' / 'graphiti'
OUT_DIR = NACRE / 'crates' / 'nacre-core' / 'tests' / 'fixtures' / 'prompts'


def stub_pydantic() -> None:
    """Install a minimal pydantic stand-in (prompt modules only need names)."""
    mod = types.ModuleType('pydantic')

    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(*args, **kwargs):  # noqa: N802 - mirrors pydantic's name
        return None

    mod.BaseModel = BaseModel
    mod.Field = Field
    sys.modules['pydantic'] = mod


def fake_package(name: str, path: Path) -> None:
    """Register `name` as a package rooted at `path` without running its
    __init__.py (upstream __init__ files pull in heavy dependencies)."""
    spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
    spec.submodule_search_locations = [str(path)]
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module


def load_prompt_module(name: str):
    stub_pydantic()
    core = GRAPHITI / 'graphiti_core'
    fake_package('graphiti_core', core)
    fake_package('graphiti_core.utils', core / 'utils')
    fake_package('graphiti_core.prompts', core / 'prompts')
    return importlib.import_module(f'graphiti_core.prompts.{name}')


# A context deliberately exercising the tricky rendering paths: bare
# f-string interpolation of lists/dicts (Python str()/repr), json.dumps
# with default separators, unicode preservation, quotes inside strings,
# optional sections, and the multi-episode attribution suffix.
ENTITY_TYPES = [
    {
        'entity_type_id': 0,
        'entity_type_name': 'Entity',
        'entity_type_description': 'Default entity classification. Use this entity type if the entity is not one of the other listed types.',
    },
    {
        'entity_type_id': 1,
        'entity_type_name': 'Person',
        'entity_type_description': "A human being mentioned by name, e.g. \"Nisha's dad\" or 佐藤さん.",
    },
]

PREVIOUS_EPISODES = [
    {'content': 'Mina: The kiln room opened last month.', 'timestamp': '2026-07-01T09:30:00+00:00'},
    {'content': 'Owen: Jordan još vodi radionicu — 陶芸クラス.', 'timestamp': None},
]

EPISODE_CONTENT = (
    'Jordan: We just moved to Denver last month. My spouse started a new role at '
    "Lockheed Martin and I enrolled in a ceramics workshop at the Belmont Arts Center."
)

ATTRIBUTION = (
    '\n7. **Episode Attribution**: The content contains multiple episodes labeled '
    '[Episode 0], [Episode 1], etc. Each episode header includes a timestamp indicating '
    'when that episode occurred. For each extracted entity, set `episode_indices` '
    'to the 0-based list of episode numbers where that entity is mentioned. '
    'An entity appearing in Episodes 0 and 2 should have `episode_indices: [0, 2]`.'
)

NODE = {
    'name': 'Jordan Lee',
    'entity_types': ['Entity', 'Person'],
    'attributes': {'phones': '415-555-0142', 'industry': None},
}

ENTITIES = [
    {'name': 'Jordan Lee', 'summary': 'Jordan Lee works at Belmont Arts Center.', 'entity_types': ['Person'], 'attributes': {}},
    {'name': 'Belmont Arts Center', 'summary': '', 'entity_types': ['Entity'], 'attributes': {}},
]

EXTRACTED_ENTITIES = [
    {'id': 0, 'name': 'Jordan'},
    {'id': 1, 'name': "Nisha's dad"},
]

ENTITY_TYPE_DESCRIPTIONS = {
    'Person': 'A human being mentioned by name.',
    'Entity': 'Default entity classification.',
}

NODES = [
    {'id': 0, 'name': 'Jordan Lee', 'entity_types': ['Person']},
    {'id': 1, 'name': 'Belmont Arts Center', 'entity_types': ['Entity']},
    {'id': 2, 'name': 'Gamecube', 'entity_types': ['Entity']},
]

EDGE_TYPES = [
    {'fact_type_id': 0, 'fact_type_name': 'WORKS_AT', 'fact_type_signature': ['Person', 'Entity'], 'fact_type_description': 'Employment relationship.'},
]

FACTS = [
    {'fact': 'Jordan Lee enrolled in a ceramics workshop last month.', 'reference_time': '2026-07-09T12:00:00Z'},
    {'fact': 'Jordan Lee no longer teaches on Fridays.', 'reference_time': '2026-07-10T08:00:00Z'},
]

EXISTING_NODES = [
    {'candidate_id': 0, 'name': 'Jordan Lee', 'entity_types': ['Person'], 'summary': 'Jordan Lee works at Belmont Arts Center.'},
    {'candidate_id': 1, 'name': 'Belmont Arts Center', 'entity_types': ['Entity'], 'summary': ''},
]

CONTEXTS: dict[str, list[dict]] = {
    'summarize_nodes': [
        {
            '_function': 'summarize_pair',
            'node_summaries': ['Jordan Lee works at Belmont Arts Center.', 'Jordan supervises two studio assistants — 陶芸の先生.'],
        },
        {
            '_function': 'summarize_context',
            'previous_episodes': PREVIOUS_EPISODES,
            'episode_content': EPISODE_CONTENT,
            'node_name': 'Jordan Lee',
            'node_summary': 'Jordan Lee works at Belmont Arts Center.',
            'attributes': ['role', 'industry'],
        },
        {
            '_function': 'summary_description',
            'summary': 'Jordan Lee teaches ceramics and supervises two assistants.',
        },
    ],
    'dedupe_nodes': [
        {
            '_function': 'node',
            'previous_episodes': PREVIOUS_EPISODES,
            'episode_content': EPISODE_CONTENT,
            'extracted_node': {'name': 'Jordan', 'entity_types': ['Person'], 'summary': ''},
            'entity_type_description': 'A human being mentioned by name.',
            'existing_nodes': EXISTING_NODES,
        },
        {
            '_function': 'nodes',
            'previous_episodes': PREVIOUS_EPISODES,
            'episode_content': EPISODE_CONTENT,
            'extracted_nodes': [
                {'id': 0, 'name': 'Jordan', 'entity_types': ['Person']},
                {'id': 1, 'name': 'Denver', 'entity_types': ['Entity']},
                {'id': 2, 'name': "Nisha's dad", 'entity_types': ['Person']},
            ],
            'existing_nodes': EXISTING_NODES,
        },
        {
            '_function': 'node_list',
            'nodes': [
                {'uuid': 'a1', 'name': 'NYC', 'summary': 'New York City'},
                {'uuid': 'b2', 'name': 'New York City — ニューヨーク', 'summary': 'The city of New York'},
            ],
        },
    ],
    'dedupe_edges': [
        {
            '_function': 'resolve_edge',
            'existing_edges': [
                {'idx': 0, 'fact': 'Jordan Lee works at Belmont Arts Center.'},
                {'idx': 1, 'fact': "Jordan Lee teaches beginner ceramics on Wednesday evenings."},
            ],
            'edge_invalidation_candidates': [
                {'idx': 2, 'fact': 'Jordan Lee lives in Denver.'},
            ],
            'new_edge': 'Jordan Lee supervises two studio assistants.',
        },
    ],
    'extract_edges': [
        {
            '_function': 'edge',
            'previous_episodes': PREVIOUS_EPISODES,
            'episode_content': EPISODE_CONTENT,
            'nodes': NODES,
            'reference_time': '2026-07-09T12:00:00Z',
            'custom_extraction_instructions': '',
        },
        {
            '_function': 'edge',
            '_case': 'with_fact_types',
            'previous_episodes': [],
            'episode_content': EPISODE_CONTENT,
            'nodes': NODES,
            'reference_time': '2026-07-09T12:00:00Z',
            'custom_extraction_instructions': 'Prefer employment facts.',
            'edge_types': EDGE_TYPES,
        },
        {
            '_function': 'extract_attributes',
            'fact': 'Jordan Lee works at Belmont Arts Center — 陶芸の先生.',
            'reference_time': '2026-07-09T12:00:00Z',
            'existing_attributes': {'role': 'teacher', 'since': None},
        },
        {
            '_function': 'extract_timestamps',
            'fact': 'Jordan Lee enrolled in a ceramics workshop last month.',
            'reference_time': '2026-07-09T12:00:00Z',
        },
        {
            '_function': 'extract_timestamps_batch',
            'facts': FACTS,
        },
    ],
    'extract_nodes': [
        {
            '_function': 'extract_message',
            'entity_types': ENTITY_TYPES,
            'previous_episodes': PREVIOUS_EPISODES,
            'episode_content': EPISODE_CONTENT,
            'custom_extraction_instructions': '',
        },
        {
            '_function': 'extract_message',
            '_case': 'with_attribution',
            'entity_types': ENTITY_TYPES,
            'previous_episodes': [],
            'episode_content': '[Episode 0] (2026-07-09T12:00:00+00:00)\nJordan: hello\n[Episode 1] (2026-07-10T12:00:00+00:00)\nMina: bye',
            'custom_extraction_instructions': 'Focus on people.' + ATTRIBUTION,
        },
        {
            '_function': 'extract_json',
            'entity_types': ENTITY_TYPES,
            'source_description': 'CRM export — 顧客データ',
            'episode_content': '{"user": "Jordan Lee", "company": "Acme Corp", "active": true}',
            'custom_extraction_instructions': '',
        },
        {
            '_function': 'extract_text',
            'entity_types': ENTITY_TYPES,
            'episode_content': 'Dr. Amara Osei presented her migraine study at the AAN conference.',
            'custom_extraction_instructions': 'Prefer full names.',
        },
        {
            '_function': 'classify_nodes',
            'previous_episodes': PREVIOUS_EPISODES,
            'episode_content': EPISODE_CONTENT,
            'extracted_entities': EXTRACTED_ENTITIES,
            'entity_types': ENTITY_TYPES,
        },
        {
            '_function': 'extract_attributes',
            'previous_episodes': PREVIOUS_EPISODES,
            'episode_content': EPISODE_CONTENT,
            'node': NODE,
        },
        {
            '_function': 'extract_summary',
            'previous_episodes': PREVIOUS_EPISODES,
            'episode_content': EPISODE_CONTENT,
            'node': NODE,
        },
        {
            '_function': 'extract_summaries_batch',
            'previous_episodes': PREVIOUS_EPISODES,
            'episode_content': EPISODE_CONTENT,
            'entities': ENTITIES,
        },
        {
            '_function': 'extract_summaries_batch',
            '_case': 'with_type_descriptions',
            'previous_episodes': PREVIOUS_EPISODES,
            'episode_content': EPISODE_CONTENT,
            'entities': ENTITIES,
            'entity_type_descriptions': ENTITY_TYPE_DESCRIPTIONS,
        },
        {
            '_function': 'extract_entity_summaries_from_episodes',
            'previous_episodes': PREVIOUS_EPISODES,
            'episode_content': EPISODE_CONTENT,
            'entities': ENTITIES,
        },
        {
            '_function': 'extract_entity_summaries_from_episodes',
            '_case': 'with_type_descriptions',
            'previous_episodes': PREVIOUS_EPISODES,
            'episode_content': EPISODE_CONTENT,
            'entities': ENTITIES,
            'entity_type_descriptions': ENTITY_TYPE_DESCRIPTIONS,
        },
    ],
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for module_name, cases in CONTEXTS.items():
        module = load_prompt_module(module_name)
        rendered = []
        for case in cases:
            context = {k: v for k, v in case.items() if not k.startswith('_')}
            function = getattr(module, case['_function'])
            messages = function(context)
            # Apply prompt_library's VersionWrapper mutation (prompts/lib.py):
            # every SYSTEM message gets DO_NOT_ESCAPE_UNICODE appended at
            # render time. All pipeline call sites go through the library,
            # so the rendered form is the ground truth to pin.
            helpers = importlib.import_module('graphiti_core.prompts.prompt_helpers')
            for m in messages:
                if m.role == 'system':
                    m.content += helpers.DO_NOT_ESCAPE_UNICODE
            rendered.append(
                {
                    'function': case['_function'],
                    'case': case.get('_case', 'default'),
                    'context': context,
                    'messages': [{'role': m.role, 'content': m.content} for m in messages],
                }
            )
        out = OUT_DIR / f'{module_name}.json'
        out.write_text(json.dumps(rendered, ensure_ascii=False, indent=1) + '\n')
        print(f'wrote {out.relative_to(NACRE)} ({len(rendered)} cases)')


if __name__ == '__main__':
    main()
