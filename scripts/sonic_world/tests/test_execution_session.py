from __future__ import annotations

import unittest

from sonic_world.skills import DispatchExecutionSession


def _action(index: int) -> dict:
    return {
        "action_id": f"action_{index}",
        "kind": "dispatch",
        "handler": "contact_grasp_primitive",
        "source_id": f"skill.{index}",
        "metadata": {"readiness": "ready", "contract": {"ready": True}},
        "command": {"type": "runtime_phase_sequence"},
    }


def _plan(count: int = 2) -> dict:
    actions = [_action(index) for index in range(count)]
    return {
        "task_id": "test_task",
        "objective": "test",
        "actions": actions,
        "next_action": actions[0],
        "metadata": {"plan_id": "test_plan"},
    }


class DispatchExecutionSessionTest(unittest.TestCase):
    def test_session_executes_strictly_in_order(self) -> None:
        session = DispatchExecutionSession(require_effect_evidence=True)
        self.assertEqual(session.load(_plan()).kind, "plan_loaded")
        self.assertEqual(session.next_action()["action_id"], "action_0")
        self.assertEqual(session.mark_dispatched("action_0").kind, "action_dispatched")
        self.assertIsNone(session.next_action())
        self.assertEqual(
            session.observe_status({"action_id": "action_0", "status": "accepted"}).kind,
            "action_feedback",
        )
        transition = session.observe_status(
            {"action_id": "action_0", "status": "success", "effect_evidence": {"passed": True}}
        )
        self.assertEqual(transition.kind, "action_succeeded")
        self.assertEqual(session.next_action()["action_id"], "action_1")
        session.mark_dispatched("action_1")
        transition = session.observe_status(
            {"action_id": "action_1", "status": "success", "effect_evidence": True}
        )
        self.assertEqual(transition.kind, "plan_succeeded")
        self.assertEqual(session.status, "succeeded")

    def test_session_rejects_success_without_effect_evidence(self) -> None:
        session = DispatchExecutionSession(require_effect_evidence=True)
        session.load(_plan(1))
        session.mark_dispatched("action_0")
        transition = session.observe_status({"action_id": "action_0", "status": "success"})
        self.assertEqual(transition.kind, "action_failed")
        self.assertIn("effect evidence", transition.reason)
        self.assertEqual(session.status, "failed")

    def test_session_is_idempotent_and_ignores_late_status(self) -> None:
        session = DispatchExecutionSession()
        session.load(_plan(1))
        self.assertEqual(session.load(_plan(1)).kind, "duplicate_plan")
        session.mark_dispatched("action_0")
        session.observe_status({"action_id": "action_0", "status": "success"})
        self.assertEqual(
            session.observe_status({"action_id": "action_0", "status": "success"}).kind,
            "duplicate_status",
        )

    def test_session_timeout_is_terminal(self) -> None:
        now = [10.0]
        session = DispatchExecutionSession(timeout_s=2.0, clock=lambda: now[0])
        session.load(_plan(1))
        session.mark_dispatched("action_0")
        now[0] = 12.1
        transition = session.check_timeout()
        self.assertIsNotNone(transition)
        self.assertEqual(transition.kind, "action_failed")
        self.assertEqual(transition.status, "timeout")

    def test_recovery_replan_can_replace_deterministic_plan(self) -> None:
        session = DispatchExecutionSession()
        session.load(_plan(1))
        session.mark_dispatched("action_0")
        session.observe_status({"action_id": "action_0", "status": "failed", "detail": "missed target"})
        transition = session.release_for_replan("navigation micro-adjust completed")
        self.assertEqual(transition.kind, "plan_released_for_replan")
        self.assertEqual(session.status, "idle")
        self.assertEqual(session.load(_plan(1)).kind, "plan_loaded")

    def test_running_action_does_not_expose_failed_action_for_recovery_replacement(self) -> None:
        session = DispatchExecutionSession()
        session.load(_plan(1))
        session.mark_dispatched("action_0")
        self.assertEqual("running", session.status)
        self.assertIsNone(session.failed_action_id)


if __name__ == "__main__":
    unittest.main()
