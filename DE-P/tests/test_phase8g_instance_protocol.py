from dataclasses import replace
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from policy.dynamic.dynamic_perception import DynamicPerception
from policy.dynamic.instance_evaluation import InstanceEvaluationState
from policy.dynamic.types import DynamicPerceptionResult
from tests.dynamic_helpers import dynamic_track, observation
from tools import phase8g_blind_protocol as blind_protocol
from tools.generate_phase8c_scenario_matrix import actor_id_base
from tools.run_phase8g_instance_recording import validate_scenario_identity_schema


def result(observations=(), tracks=()):
    tracks = tuple(tracks)
    return DynamicPerceptionResult(
        all_tracks=tracks,
        confirmed_tracks=tuple(x for x in tracks if x.is_confirmed),
        dynamic_tracks=tuple(
            x for x in tracks if x.is_dynamic and x.attention_authorized
        ),
        projected_dynamic_tracks=(),
        attention_map=torch.zeros(1, 1, 3, 5),
        observations=tuple(observations),
    )


def actor(actor_id, position, pixels, visible=True):
    return {
        "object_id": actor_id, "position_world": list(position), "radius": 0.4,
        "visible": visible, "rendered_pixel_count": pixels,
    }


def grounded(frame=0, actor_pixels=(10, 11, 12), track_id=0):
    obs = replace(
        observation([0, 0, 4], timestamp=float(frame)),
        observation_id=frame, frame_index=frame,
        direct_image_evidence=True, pixel_indices=actor_pixels,
        pixel_bbox=(0, 0, 2, 0), component_pixel_count=len(actor_pixels),
    )
    track = replace(
        dynamic_track(track_id=track_id, position=[0, 0, 4], dynamic=True),
        is_confirmed=True, birth_observation_id=frame, birth_frame=frame,
        last_observation_id=frame, last_direct_observation_frame=frame,
        direct_observation_count=1, consecutive_direct_hits=1,
        ever_directly_observed=True, attention_authorized=True,
    )
    return obs, track


class Phase8GInstanceProtocolTest(unittest.TestCase):
    def test_visible_component_is_grounded_by_instance_overlap(self):
        state = InstanceEvaluationState()
        instance = np.zeros((3, 5), np.int32)
        instance.reshape(-1)[[10, 11, 12]] = 7
        obs, track = grounded()
        state.update("visible", 0, [actor(7, [0, 0, 4], 3)],
                     instance, result([obs], [track]))
        summary = state.summary()
        self.assertEqual(summary["track_birth_instance_precision"], 1.0)
        self.assertEqual(summary["visible_instance_recall"], 1.0)

    def test_never_observed_near_other_track_is_proximity_not_identity(self):
        state = InstanceEvaluationState()
        instance = np.zeros((3, 5), np.int32)
        obs, track = grounded(actor_pixels=(0, 1, 2))
        state.update("negative", 0, [actor(9, [0, 0, 4.1], 0, visible=False)],
                     instance, result([obs], [track]))
        summary = state.summary()
        self.assertEqual(summary["never_observed_proximity_event_count"], 1)
        self.assertEqual(summary["visible_actor_matches"], 0)
        self.assertEqual(summary["never_observed_attention_event_count"], 1)

    def test_prediction_only_confirmation_and_attention_are_hard_failures(self):
        state = InstanceEvaluationState()
        track = replace(
            dynamic_track(track_id=3, position=[0, 0, 4], dynamic=True),
            is_confirmed=True, attention_authorized=True,
            ever_directly_observed=False, direct_observation_count=0,
        )
        state.update("ungrounded", 0, [], np.zeros((3, 5), np.int32),
                     result([], [track]))
        summary = state.summary()
        self.assertEqual(summary["prediction_only_confirmed_track_count"], 1)
        self.assertEqual(summary["prediction_only_attention_count"], 1)
        self.assertEqual(summary["never_observed_ungrounded_track_count"], 1)

    def test_instance_count_disagreement_is_illegal_data(self):
        state = InstanceEvaluationState()
        instance = np.zeros((3, 5), np.int32)
        instance[0, 0] = 5
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            state.update("bad", 0, [actor(5, [0, 0, 4], 2)],
                         instance, result())

    def test_two_close_actors_are_assigned_by_pixel_identity(self):
        state = InstanceEvaluationState()
        instance = np.zeros((3, 5), np.int32)
        instance.reshape(-1)[[0, 1, 2]] = 4
        instance.reshape(-1)[[10, 11, 12]] = 5
        obs_a, track_a = grounded(actor_pixels=(0, 1, 2), track_id=10)
        obs_b, track_b = grounded(actor_pixels=(10, 11, 12), track_id=11)
        obs_b = replace(obs_b, observation_id=1)
        track_b = replace(track_b, last_observation_id=1, birth_observation_id=1)
        actors = [
            actor(4, [0.0, 0.0, 4.0], 3),
            actor(5, [0.05, 0.0, 4.0], 3),
        ]
        state.update("close", 0, actors, instance,
                     result([obs_a, obs_b], [track_a, track_b]))
        identities = {
            event["track_id"]: event["actor_id"] for event in state.events
            if event["classification_reason"] == "direct_instance_overlap"
        }
        self.assertEqual(identities, {10: 4, 11: 5})

    def test_runtime_perception_api_has_no_instance_mask_input(self):
        parameters = inspect.signature(DynamicPerception.update_depth).parameters
        self.assertFalse(any("instance" in name for name in parameters))

    def test_blind_lock_refuses_a_second_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / ".phase8g_blind_once.lock"
            lock.write_text("{}")
            with mock.patch.object(blind_protocol, "REPORTS", root), \
                    mock.patch.object(blind_protocol, "LOCK", lock):
                with self.assertRaises(FileExistsError):
                    blind_protocol.assert_one_shot_targets_absent(
                        root / "blind", root / "maps"
                    )

    def test_blind_hash_enforcement_detects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.py"
            development = root / "development.json"
            shadow = root / "shadow.json"
            source.write_text("frozen")
            development.write_text('{"status":"PASS"}')
            shadow.write_text('{"status":"PASS"}')
            sources = [source]
            reports = [development, shadow]
            frozen = blind_protocol.frozen_hashes(sources, reports, 12345)
            source.write_text("mutated")
            with self.assertRaises(RuntimeError):
                blind_protocol.assert_frozen(frozen, sources, reports, 12345)

    def test_historical_phase8f_test_rerun_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "audit-only"):
            blind_protocol.assert_not_historical_dataset(
                blind_protocol.HISTORICAL_CONSUMED_TEST
            )

    def test_large_valid_actor_ids_fit_ros_32sc1_without_collision(self):
        seeds = range(536_870_890, 536_870_904)
        ids = {
            actor_id_base(seed) + slot
            for seed in seeds for slot in (1, 2, 3)
        }
        self.assertEqual(len(ids), len(tuple(seeds)) * 3)
        self.assertLessEqual(max(ids), 2**31 - 1)

    def test_actor_id_encoding_rejects_uint32_overflow(self):
        with self.assertRaisesRegex(ValueError, "cannot be encoded"):
            actor_id_base(536_870_912)

    def test_recording_preflight_rejects_instance_id_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            scenario = Path(directory) / "overflow.yaml"
            scenario.write_text(
                "dynamic_scenario:\n"
                "  seed: 1\n"
                "  actors:\n"
                "  - id: 2147483648\n"
            )
            with self.assertRaisesRegex(ValueError, "exceeds ROS 32SC1"):
                validate_scenario_identity_schema([{
                    "sequence_id": "overflow",
                    "scenario_file": str(scenario),
                }])
