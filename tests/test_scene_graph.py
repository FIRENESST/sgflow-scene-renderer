import json

import pytest
import torch

from sgflow.scene_graph import SceneGraph


CATEGORIES = ["pad", "chair", "table"]


def test_versioned_json_round_trip_and_fingerprint():
    graph = SceneGraph.from_objects(
        [{"category": "chair", "position": [1, 2, 3], "scale": [1, 2, 3], "appearance": [0.2, 0.3]}], CATEGORIES, 2
    )
    serialized = json.dumps(graph.to_dict())
    assert json.loads(serialized)["schema_version"] == 1
    restored = SceneGraph.from_dict(json.loads(serialized), CATEGORIES, 2)
    assert torch.equal(restored.cat, graph.cat)
    assert torch.allclose(restored.to_latent(), graph.to_latent())
    assert restored.fingerprint() == graph.fingerprint()


def test_legacy_json_and_empty_scene_shapes():
    graph = SceneGraph.from_dict(json.loads('{"objects": []}'), CATEGORIES, 4)
    assert graph.cat.shape == (0,)
    assert graph.pos.shape == (0, 3)
    assert graph.rot6d.shape == (0, 6)
    assert graph.log_scale.shape == (0, 3)
    assert graph.appearance.shape == (0, 4)
    assert graph.morton_sorted().n == 0


@pytest.mark.parametrize(("objects", "message"), [
    ([{"category": "pad", "position": [0, 0, 0]}], "object 0: category"),
    ([{"category": "chair", "position": [0, 0]}], "object 0: position"),
    ([{"category": "chair", "position": [0, 0, 0], "scale": [1, 0, 1]}], "object 0: scale"),
    ([{"category": "chair", "position": [0, float("nan"), 0]}], "object 0: position"),
])
def test_invalid_object_has_clear_index(objects, message):
    with pytest.raises(ValueError, match=message):
        SceneGraph.from_objects(objects, CATEGORIES)


def test_tensor_shape_and_schema_validation():
    with pytest.raises(ValueError, match="contains PAD"):
        SceneGraph(CATEGORIES, torch.tensor([0]), torch.zeros(1, 3), torch.zeros(1, 6), torch.zeros(1, 3), torch.zeros(1, 2))
    with pytest.raises(ValueError, match="unsupported schema_version"):
        SceneGraph.from_dict({"schema_version": 2, "objects": []}, CATEGORIES)
