import hashlib
from coordination.models.node import Node
from coordination.models.task import Task, TaskStatus
from coordination.config import settings


def hash_result(result) -> str:
    """Compute a canonical SHA-256 hash of a task result."""
    import json
    serialized = json.dumps(result, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def verify_challenge(task: Task, submitted_hash: str) -> bool:
    """
    Verify a challenge task result against the known answer.
    Returns True if the submitted hash matches the expected answer.
    """
    if not task.is_challenge or task.challenge_answer is None:
        return True  # Not a challenge task, skip verification
    return submitted_hash == task.challenge_answer


def check_redundancy_agreement(
    task: Task,
    sibling_tasks: list[Task],
    submitted_hash: str
) -> tuple[bool, float]:
    """
    Compare submitted result against sibling task results.
    Returns (agreement: bool, consistency_score: float in [0, 1]).
    """
    completed_siblings = [
        t for t in sibling_tasks
        if t.status in (TaskStatus.COMPLETED, TaskStatus.VERIFIED)
        and t.result_hash is not None
    ]
    if not completed_siblings:
        return True, 1.0  # No siblings yet, cannot verify

    matches = sum(1 for t in completed_siblings if t.result_hash == submitted_hash)
    consistency_score = matches / len(completed_siblings)
    agreed = consistency_score >= 0.5
    return agreed, consistency_score


def update_reliability_factor(node: Node, consistency_score: float) -> float:
    """
    Recompute reliability factor after a task result.
    R = 0.5 + 0.5 × (C_completed / C_assigned × V_consistency)
    """
    completion_rate = node.completion_rate
    new_reliability = 0.5 + 0.5 * (completion_rate * consistency_score)
    new_reliability = max(settings.RELIABILITY_MIN,
                          min(settings.RELIABILITY_MAX, new_reliability))
    return round(new_reliability, 4)


def should_suspend(node: Node) -> bool:
    """
    Returns True if a node's reliability factor has fallen below
    the suspension threshold AND it has completed enough tasks
    to make the assessment reliable.
    """
    return (
        node.tasks_completed >= settings.RELIABILITY_MIN_TASKS_BEFORE_SUSPEND and
        node.reliability_factor < settings.RELIABILITY_SUSPEND_THRESHOLD
    )


def get_redundancy_rate(node: Node) -> float:
    """
    Determine the redundancy verification rate for a node.
    New nodes receive higher redundancy; trusted nodes lower.
    """
    if node.tasks_completed < 20:
        return settings.REDUNDANCY_RATE_NEW_NODE
    if node.reliability_factor >= 0.95:
        return settings.REDUNDANCY_RATE_TRUSTED_NODE
    # Linear interpolation between new and trusted rates
    trust_progress = min(1.0, (node.tasks_completed - 20) / 80)
    rate = (settings.REDUNDANCY_RATE_NEW_NODE +
            trust_progress * (settings.REDUNDANCY_RATE_TRUSTED_NODE -
                               settings.REDUNDANCY_RATE_NEW_NODE))
    return round(rate, 4)
