"""Contract tests for the Go1 motion_feedback sensor card."""

import importlib.util
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
GO1_DIR = ROOT / "unitree/go1"
sys.path.insert(0, str(GO1_DIR))

import go1_sdk_client  # noqa: E402
import sensors  # noqa: E402


CONFIG = sensors._motion_feedback_config({})


def snapshot(*, command=None, velocity=(0.0, 0.0, 0.0), yaw_speed=0.0, fresh=True):
    if command is None:
        command = {"active": False}
    return {
        "fresh": fresh,
        "control_level": "HIGHLEVEL",
        "telemetry_age_sec": 0.01,
        "commanded_motion": command,
        "velocity": list(velocity) if velocity is not None else None,
        "yaw_speed": yaw_speed,
    }


def active_command(vx=0.3, vy=0.0, vyaw=0.0, active_for_sec=1.0):
    return {
        "active": True,
        "vx": vx,
        "vy": vy,
        "vyaw": vyaw,
        "gait": 1,
        "active_for_sec": active_for_sec,
    }


class MotionFeedbackBuilderTests(unittest.TestCase):
    def test_idle_without_active_command(self):
        payload = sensors._build_motion_feedback(snapshot(), CONFIG)

        self.assertEqual(payload["status"], "idle")
        self.assertEqual(payload["reason"], "no_active_motion_command")
        self.assertEqual(payload["recommended_action"], "none")

    def test_starting_during_grace_period(self):
        payload = sensors._build_motion_feedback(
            snapshot(command=active_command(active_for_sec=0.1)), CONFIG
        )

        self.assertEqual(payload["status"], "starting")
        self.assertEqual(payload["recommended_action"], "wait")

    def test_forward_motion_must_match_command_direction(self):
        moving = sensors._build_motion_feedback(
            snapshot(command=active_command(), velocity=(0.12, 0.0, 0.0)), CONFIG
        )
        opposite = sensors._build_motion_feedback(
            snapshot(command=active_command(), velocity=(-0.12, 0.0, 0.0)), CONFIG
        )

        self.assertEqual(moving["status"], "moving")
        self.assertEqual(moving["recommended_action"], "continue")
        self.assertEqual(opposite["status"], "motion_not_observed")
        self.assertLess(opposite["observed"]["along_command_mps"], 0.0)

    def test_yaw_motion_must_match_command_direction(self):
        left = sensors._build_motion_feedback(
            snapshot(
                command=active_command(vx=0.0, vyaw=0.5),
                yaw_speed=0.2,
            ),
            CONFIG,
        )
        wrong_way = sensors._build_motion_feedback(
            snapshot(
                command=active_command(vx=0.0, vyaw=0.5),
                yaw_speed=-0.2,
            ),
            CONFIG,
        )

        self.assertEqual(left["status"], "moving")
        self.assertEqual(wrong_way["status"], "motion_not_observed")

    def test_accepted_command_without_motion_is_actionable(self):
        payload = sensors._build_motion_feedback(
            snapshot(command=active_command(active_for_sec=0.8)), CONFIG
        )

        self.assertEqual(payload["status"], "motion_not_observed")
        self.assertEqual(payload["reason"], "accepted_command_without_measured_motion")
        self.assertEqual(payload["recommended_action"], "stop_and_inspect")

    def test_stale_or_incomplete_input_is_unavailable(self):
        cases = [
            (snapshot(fresh=False), "telemetry_stale"),
            (snapshot() | {"telemetry_age_sec": 1.0}, "telemetry_stale"),
            ({"fresh": True, "telemetry_age_sec": 0.01}, "command_state_missing"),
            (
                snapshot(command=active_command(), velocity=None),
                "velocity_missing",
            ),
            (
                snapshot(command={"active": True, "vx": "bad"}),
                "command_state_invalid",
            ),
        ]
        for snap, reason in cases:
            with self.subTest(reason=reason):
                payload = sensors._build_motion_feedback(snap, CONFIG)
                self.assertEqual(payload["status"], "unavailable")
                self.assertFalse(payload["available"])
                self.assertEqual(payload["reason"], reason)

    def test_near_zero_command_is_not_a_false_alarm(self):
        payload = sensors._build_motion_feedback(
            snapshot(command=active_command(vx=0.01, vyaw=0.01)), CONFIG
        )

        self.assertEqual(payload["status"], "idle")
        self.assertEqual(payload["reason"], "no_effective_motion_command")


class MotionFeedbackPluginTests(unittest.TestCase):
    def test_plugin_contract_is_read_only_data_json(self):
        class Client:
            def snapshot(self):
                return snapshot()

        plugin = sensors.MotionFeedbackPlugin({}, "go1", None, Client())
        tool = plugin.get_tool()

        self.assertEqual(tool["name"], "motion_feedback")
        self.assertEqual(tool["type"], "sensor")
        self.assertEqual(tool["inputSchema"], {"type": "object", "properties": {}})
        self.assertEqual(plugin.dispatch("read", {})["data"]["status"], "idle")
        self.assertIsNone(plugin.dispatch("unknown", {}))
        plugin._node = object()
        self.assertEqual(
            plugin.get_tool()["topic_out"],
            [{"topic": "/go1/state/motion_feedback", "format": "data/json"}],
        )

    def test_invalid_configuration_fails_fast(self):
        cases = [
            {"publish_hz": 0},
            {"publish_hz": 21},
            {"startup_grace_sec": -1},
            {"max_telemetry_age_sec": 0},
            {"min_linear_command_mps": "bad"},
            {"min_yaw_response_rad_s": float("nan")},
        ]
        for config in cases:
            with self.subTest(config=config), self.assertRaises(ValueError):
                sensors.MotionFeedbackPlugin(config, "go1", None, object())

    def test_manifest_config_and_bundle_register_the_card(self):
        manifest = (GO1_DIR / "driver.yaml").read_text()
        config = (GO1_DIR / "config.yaml").read_text()
        main = (GO1_DIR / "main.py").read_text()

        self.assertIn("- { name: motion_feedback,    type: sensor }", manifest)
        self.assertIn("  motion_feedback:", config)
        self.assertIn("sensors.make_motion_feedback", main)

        yaml_stub = types.ModuleType("yaml")
        yaml_stub.safe_load = lambda value: value
        spec = importlib.util.spec_from_file_location(
            "go1_motion_feedback_main_test", GO1_DIR / "main.py"
        )
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"yaml": yaml_stub}):
            spec.loader.exec_module(module)

        class Client:
            def snapshot(self):
                return snapshot()

        bundle = module.Go1Bundle(
            {"plugins": {"motion_feedback": {"enabled": True}}},
            "go1",
            None,
            Client(),
        )
        self.assertEqual(
            [tool["name"] for tool in bundle.get_all_tools()], ["motion_feedback"]
        )
        self.assertEqual(bundle.dispatch("motion_feedback", {})["data"]["status"], "idle")


class CommandTrackingTests(unittest.TestCase):
    @staticmethod
    def make_client():
        client = go1_sdk_client.Go1HighSdkClient.__new__(
            go1_sdk_client.Go1HighSdkClient
        )
        client._lock = threading.Lock()
        client._snapshot = {
            "fresh": True,
            "control_level": "HIGHLEVEL",
            "velocity": [0.0, 0.0, 0.0],
            "yaw_speed": 0.0,
        }
        client._snapshot_received_at = 9.9
        client._move_cmd = None
        client._move_deadline = 0.0
        client._move_started_at = 0.0
        client._posture = None
        client._desired_gait = 1
        return client

    def test_repeated_same_command_preserves_start_time(self):
        client = self.make_client()
        with mock.patch.object(
            go1_sdk_client.time, "monotonic", side_effect=[10.0, 10.1, 10.2]
        ):
            client.move(vx=0.3)
            client.move(vx=0.3)
            snap = client.snapshot()

        self.assertTrue(snap["commanded_motion"]["active"])
        self.assertEqual(snap["commanded_motion"]["active_for_sec"], 0.2)

    def test_changed_command_restarts_timer_and_stop_clears_it(self):
        client = self.make_client()
        with mock.patch.object(
            go1_sdk_client.time, "monotonic", side_effect=[10.0, 10.2, 10.3]
        ):
            client.move(vx=0.3)
            client.move(vx=-0.3)
            snap = client.snapshot()

        self.assertEqual(snap["commanded_motion"]["active_for_sec"], 0.1)
        client.stop_move()
        with mock.patch.object(go1_sdk_client.time, "monotonic", return_value=10.4):
            self.assertEqual(client.snapshot()["commanded_motion"], {"active": False})


if __name__ == "__main__":
    unittest.main()
