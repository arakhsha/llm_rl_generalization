import os
from pathlib import Path

import pytest
import textworld
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

from llm_rl.alfworld import (
    AlfworldEnv,
    alfworld_reward,
    clean_observation,
    extract_games,
    load_manifest,
    normalize_command,
)
from llm_rl.env_pool import EnvPool

DATA_DIR = os.environ.get("ALFWORLD_DATA", "")
needs_data = pytest.mark.skipif(
    not (Path(DATA_DIR) / "manifest.jsonl").exists(), reason="ALFWorld games not packed yet (setup_alfworld.sbatch)"
)


def test_normalize_command():
    assert normalize_command("  Go to  Countertop 1. ") == "go to countertop 1"
    assert normalize_command('"take apple 1 from fridge 1"') == "take apple 1 from fridge 1"


def test_clean_observation():
    text = "-= Welcome to TextWorld, ALFRED! =-\n\nYou are in the middle of a room."
    assert clean_observation(text) == "You are in the middle of a room."


@pytest.fixture(scope="module")
def games_dir(tmp_path_factory):
    return extract_games(DATA_DIR, str(tmp_path_factory.mktemp("games")))


def expert_plan(games_dir: str, game: str) -> list[str]:
    infos = textworld.EnvInfos(policy_commands=True)
    env = textworld.start(f"{games_dir}/{game}", infos, wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])
    return env.reset()["policy_commands"]


@needs_data
def test_manifest_splits():
    # Upstream ALFWorld counts for the text environment.
    assert len(load_manifest(DATA_DIR, "valid_seen")) == 140
    assert len(load_manifest(DATA_DIR, "valid_unseen")) == 134
    assert len(load_manifest(DATA_DIR, "train")) == 3553


@needs_data
def test_expert_plan_wins(games_dir):
    for row in load_manifest(DATA_DIR, "valid_unseen")[:5]:
        env = AlfworldEnv(games_dir, max_steps=50)
        first = env.reset(**row)
        assert "Your task is to:" in first
        for command in expert_plan(games_dir, row["game"]):
            obs = env.act(command)
        assert env.won and env.done and obs.endswith("Task completed.")


@needs_data
def test_invalid_commands_and_step_limit(games_dir):
    row = load_manifest(DATA_DIR, "valid_seen")[0]
    env = AlfworldEnv(games_dir, max_steps=2, admissible_commands="all")
    assert "Available commands:" in env.reset(**row)
    assert env.act("fly to the moon").startswith("Nothing happens.")
    assert env.act("look").endswith("Out of commands. The episode is over.")
    assert env.act("look") == "The episode is over. Do not call act again."
    assert env.stats() == {"won": False, "done": True, "steps": 2, "invalid": 1}


@needs_data
def test_local_admissible_commands(games_dir):
    row = load_manifest(DATA_DIR, "valid_seen")[0]
    env = AlfworldEnv(games_dir, max_steps=50, admissible_commands="local")
    listed = env.reset(**row).split("Available here besides go to, look and inventory: ")[1]
    assert "go to" not in listed and "look" not in listed.split(", ")
    assert "Available" not in AlfworldEnv(games_dir, max_steps=50, admissible_commands="none").reset(**row)


def test_alfworld_reward_logs_metrics():
    logged = {}
    rewards = alfworld_reward(
        won=[True, False, False],
        steps=[5, 50, 3],
        invalid=[0, 10, 0],
        done=[True, True, False],
        task_type=["pick_and_place_simple", "pick_and_place_simple", "look_at_obj_in_light"],
        log_metric=logged.__setitem__,
    )
    assert rewards == [1.0, 0.0, 0.0]
    assert logged["alfworld/invalid_rate"] == 10 / 58
    assert logged["alfworld/unfinished"] == 1 / 3
    assert logged["alfworld/success/pick_and_place_simple"] == 0.5


@needs_data
def test_env_pool_matches_direct_env(games_dir):
    rows = load_manifest(DATA_DIR, "valid_seen")[:3]
    pool = EnvPool(AlfworldEnv, 2, games_dir=games_dir, max_steps=50)
    try:
        resets = pool.run([(i, "reset", row) for i, row in enumerate(rows)])
        for row, (obs, stats) in zip(rows, resets, strict=True):
            assert obs == AlfworldEnv(games_dir, max_steps=50).reset(**row)
            assert stats == {"won": False, "done": False, "steps": 0, "invalid": 0}
        # Requests for one slot run in order; bad arguments come back as the exception.
        results = pool.run([(0, "act", {"command": "look"}), (0, "act", {"command": "jump"}), (1, "act", {"cmd": "x"})])
        assert results[1][1]["steps"] == 2 and results[1][0].startswith("Nothing happens.")
        assert isinstance(results[2][0], TypeError)
    finally:
        pool.close()
