import json
from types import SimpleNamespace

import pytest
import torch

from sgflow import ScenePipeline
from sgflow.config import SGFlowConfig
from sgflow.openai_compat import (
    LLMServiceError,
    OpenAICompatibleConfig,
    OpenAICompatibleScenePipeline,
)


def tiny_cfg():
    return SGFlowConfig(
        text_model="unused", text_dim=8, d_model=8, n_layers=1, n_latents=2,
        d_state=2, expand=1, max_objects=4, max_generated_objects=4,
        d_appearance=2, flow_steps=1, room_size=(4.0, 4.0, 3.0),
        categories=["pad", "chair", "table", "lamp"], use_compile=False,
    )


PLAN = {
    "objects": [
        {"id": "chair_1", "category": "chair", "position": [-0.9, 0.0, 0.5],
         "size": [0.6, 0.6, 1.0], "yaw_degrees": -90},
        {"id": "table_1", "category": "table", "position": [0.8, 0.0, 0.4],
         "size": [1.0, 0.8, 0.8], "yaw_degrees": 0},
    ],
    "relations": [
        {"subject_id": "chair_1", "relation": "left_of", "object_id": "table_1"},
        {"subject_id": "chair_1", "relation": "facing", "object_id": "table_1"},
    ],
}


class FakeCompletions:
    def __init__(self, payload=PLAN):
        self.payload = payload
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        message = SimpleNamespace(content=json.dumps(self.payload), refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class RejectedFormatError(Exception):
    status_code = 400


def fake_client(payload=PLAN):
    completions = FakeCompletions(payload)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def test_openai_compatible_pipeline_uses_schema_and_returns_bounded_scene():
    client, completions = fake_client()
    service = OpenAICompatibleConfig(
        model="local-model", base_url="http://127.0.0.1:8000/v1", api_key="test",
    )
    pipeline = OpenAICompatibleScenePipeline(service, tiny_cfg(), device="cpu", client=client)
    scene = pipeline.generate("a chair beside a table", refine_steps=4, seed=7)

    assert scene.n == 2
    assert scene.metadata["generator"] == "openai-compatible"
    assert scene.metadata["spatial_model"] == "sparse-relation-graph+obb-sat"
    assert scene.metadata["object_ids"] == ["chair_1", "table_1"]
    request = completions.requests[0]
    assert request["model"] == "local-model"
    assert request["response_format"]["type"] == "json_schema"
    assert "variation key 7" in request["messages"][-1]["content"]
    lower, upper = scene.aabb()
    assert torch.all(lower[:, :2] >= -2.0001)
    assert torch.all(upper[:, :2] <= 2.0001)
    assert torch.all(lower[:, 2] >= -1e-5)
    assert torch.all(upper[:, 2] <= 3.0001)


def test_scene_pipeline_factory_reads_explicit_openai_compatible_settings():
    client, _ = fake_client()
    pipeline = ScenePipeline.from_openai(
        model="fake", base_url="http://localhost:1234/v1", api_key="none",
        cfg=tiny_cfg(), device="cpu", client=client,
    )
    assert isinstance(pipeline, OpenAICompatibleScenePipeline)
    assert pipeline.generate("a small room", refine_steps=0).n == 2


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"objects": [], "relations": []}, "non-empty"),
        ({"objects": [{"id": "x", "category": "sofa", "position": [0, 0, 0],
                       "size": [1, 1, 1], "yaw_degrees": 0}], "relations": []}, "unsupported"),
        ({"objects": PLAN["objects"], "relations": [
            {"subject_id": "missing", "relation": "near", "object_id": "table_1"}
        ]}, "unknown subject"),
    ],
)
def test_invalid_model_plans_fail_before_scene_construction(payload, message):
    client, _ = fake_client(payload)
    pipeline = OpenAICompatibleScenePipeline(
        OpenAICompatibleConfig(model="fake", api_key="test"), tiny_cfg(),
        device="cpu", client=client,
    )
    with pytest.raises(LLMServiceError, match=message):
        pipeline.generate("anything", refine_steps=0)


def test_local_base_url_can_use_placeholder_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = OpenAICompatibleConfig.from_env(
        model="local", base_url="http://localhost:8000/v1",
    )
    assert service.api_key == "not-needed"


def test_auto_mode_falls_back_only_after_format_rejection():
    class FallbackCompletions(FakeCompletions):
        def create(self, **kwargs):
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                raise RejectedFormatError("json_schema unsupported")
            message = SimpleNamespace(content=json.dumps(PLAN), refusal=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    completions = FallbackCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    pipeline = OpenAICompatibleScenePipeline(
        OpenAICompatibleConfig(model="fake", api_key="test"), tiny_cfg(),
        device="cpu", client=client,
    )
    assert len(pipeline.plan("a room").objects) == 2
    assert completions.requests[0]["response_format"]["type"] == "json_schema"
    assert completions.requests[1]["response_format"]["type"] == "json_object"
