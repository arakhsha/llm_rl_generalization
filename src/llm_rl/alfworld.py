"""Text-only ALFWorld as a multi-turn tool environment.

Each episode plays one game: the model calls the `act` tool with one text command per turn and reads
the observation that comes back. The reward is 1.0 if the game is won, else 0.0. Episodes are driven
by llm_rl.alfworld_rollout, with the environments running in worker processes (llm_rl.env_pool.EnvPool).

Games are packed once by `python -m llm_rl.alfworld <raw_dir> <data_dir>` (see scripts/setup_alfworld.sbatch)
into `games.tar` plus `manifest.jsonl`, and each job extracts the tar to node-local disk.
"""

import fcntl
import io
import json
import re
import sys
import tarfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from statistics import mean

import textworld
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

# The six ALFWorld task families (AlfredTWEnv's TASK_TYPES).
TASK_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)
# valid_seen / valid_unseen are the usual in- and out-of-distribution evaluation sets (140 / 134 games).
SPLITS = ("train", "valid_seen", "valid_unseen")
GAMES_ZIP = "json_2.1.2_tw-pddl.zip"
TRAJ_ZIP = "json_2.1.1_json.zip"

SYSTEM_PROMPT = """\
You are an agent in ALFWorld, a text-based household environment. Complete the task given in the \
first observation by calling the `act` tool with one command at a time, reading each observation \
before choosing the next command.

Commands:
- go to <receptacle>
- open <receptacle> / close <receptacle>
- take <object> from <receptacle>
- move <object> to <receptacle>
- heat <object> with <receptacle> / cool <object> with <receptacle> / clean <object> with <receptacle>
- use <object>
- examine <object or receptacle>
- look / inventory

Use full names with their number, such as "apple 1" or "countertop 2". You can carry one object at \
a time and must go to a receptacle before using it. "Nothing happens." means the command was not \
valid. The number of commands is limited, so be efficient."""

USER_PROMPT = "Here is your first observation.\n\n"


def build_prompt() -> list[dict[str, str]]:
    """Conversation prefix; the environment's reset() appends the first observation to the user turn."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": USER_PROMPT}]


def normalize_command(command: str) -> str:
    """Lower-case, collapse whitespace and drop surrounding quotes / a trailing period."""
    command = " ".join(command.strip().strip("\"'`").split()).lower()
    return command.rstrip(".")


def clean_observation(text: str) -> str:
    """Drop TextWorld's welcome banner and surrounding whitespace."""
    return re.sub(r"^-= Welcome to TextWorld, ALFRED! =-\s*", "", text.strip())


class AlfworldEnv:
    """One ALFWorld game at a time: reset() loads a game, act() runs a command and returns the observation.

    The signature and docstring of `act` define the tool schema shown to the model. One step takes
    ~70 ms and loading a game ~1 s, so batches of episodes run in an EnvPool.
    """

    def __init__(self, games_dir: str, max_steps: int, admissible_commands: str = "local"):
        """`admissible_commands` controls the list of valid commands appended to each observation:
        "none", "all", or "local" (all except go to / look / inventory, which the model can always use;
        the receptacles to go to are named in the first observation). "all" roughly triples the tokens
        per step.
        """
        assert admissible_commands in ("none", "local", "all"), admissible_commands
        self._games_dir = Path(games_dir)
        self._max_steps = max_steps
        self._admissible = admissible_commands
        self._infos = textworld.EnvInfos(won=True, admissible_commands=True)
        self._env = None
        self._game = None
        self.won = False
        self.done = False
        self.steps = 0
        self.invalid = 0

    def reset(self, game: str, **kwargs) -> str:
        path = str(self._games_dir / game)
        if self._env is None:
            self._env = textworld.start(path, self._infos, wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])
        elif game != self._game:
            # Reuse the PddlEnv: each new one dlopens its own copy of fast-downward's library, which is never freed.
            self._env.load(path)
        self._game = game
        self._state = self._env.reset()
        self.won = self.done = False
        self.steps = self.invalid = 0
        return self._observation(clean_observation(self._state.feedback))

    def act(self, command: str) -> str:
        """Run one command in the household environment.

        Args:
            command: The command to run, e.g. "go to countertop 1" or "take apple 1 from countertop 1".

        Returns:
            The observation after the command.
        """
        if self.done:
            return "The episode is over. Do not call act again."
        self._state, _, _ = self._env.step(normalize_command(command))
        self.steps += 1
        feedback = clean_observation(self._state.feedback)
        self.invalid += feedback == "Nothing happens."
        if self._state["won"]:
            self.won = self.done = True
            return f"{feedback}\n\nTask completed."
        if self.steps >= self._max_steps:
            self.done = True
            return f"{feedback}\n\nOut of commands. The episode is over."
        return self._observation(feedback)

    def stats(self) -> dict:
        return {"won": self.won, "done": self.done, "steps": self.steps, "invalid": self.invalid}

    def _observation(self, feedback: str) -> str:
        commands = [c for c in self._state["admissible_commands"] if c != "help"]
        if self._admissible == "all":
            return f"{feedback}\n\nAvailable commands: {', '.join(commands)}"
        if self._admissible == "local":
            local = [c for c in commands if not c.startswith("go to ") and c not in ("look", "inventory")]
            return f"{feedback}\n\nAvailable here besides go to, look and inventory: {', '.join(local) or 'none'}"
        return feedback


def alfworld_reward(
    won: list[bool],
    steps: list[int],
    invalid: list[int],
    done: list[bool],
    task_type: list[str],
    log_metric: Callable[[str, float], None] | None = None,
    **kwargs,
) -> list[float]:
    """1.0 if the game was won, else 0.0. The episode fields come from the rollout function.

    Also logs episode statistics and per-task success.
    """
    rewards = [float(w) for w in won]
    if log_metric is not None:
        log_metric("alfworld/steps", mean(steps))
        log_metric("alfworld/invalid_rate", sum(invalid) / max(1, sum(steps)))
        # Episodes that stopped calling the tool (or ran out of tokens) before the game was over.
        log_metric("alfworld/unfinished", mean(not d for d in done))
        for task in sorted(set(task_type)):
            task_rewards = [r for r, t in zip(rewards, task_type, strict=True) if t == task]
            log_metric(f"alfworld/success/{task}", mean(task_rewards))
    return rewards


def build_archive(raw_dir: Path, data_dir: Path) -> None:
    """Pack the solvable games of each split into games.tar plus manifest.jsonl.

    Mirrors AlfredTWEnv.collect_game_files: skips movable/sliced trajectories, unknown task types and
    games not marked solvable.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], {"movable_or_sliced": 0, "no_traj": 0, "task_type": 0, "unsolvable": 0}
    with (
        zipfile.ZipFile(raw_dir / GAMES_ZIP) as games_zip,
        zipfile.ZipFile(raw_dir / TRAJ_ZIP) as traj_zip,
        tarfile.open(data_dir / "games.tar", "w") as tar,
    ):
        trajs = {"/".join(n.split("/")[-4:-1]): n for n in traj_zip.namelist() if n.endswith("/traj_data.json")}
        for name in sorted(games_zip.namelist()):
            if not name.endswith("/game.tw-pddl"):
                continue
            key = "/".join(name.split("/")[-4:-1])  # <split>/<task dir>/<trial dir>
            if key.split("/")[0] not in SPLITS:
                continue
            if "movable" in key or "Sliced" in key:
                skipped["movable_or_sliced"] += 1
                continue
            if key not in trajs:
                skipped["no_traj"] += 1
                continue
            task_type = json.loads(traj_zip.read(trajs[key]))["task_type"]
            if task_type not in TASK_TYPES:
                skipped["task_type"] += 1
                continue
            data = games_zip.read(name)
            if not json.loads(data).get("solvable", False):
                skipped["unsolvable"] += 1
                continue
            game = f"{key}/game.tw-pddl"
            info = tarfile.TarInfo(game)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
            rows.append({"split": key.split("/")[0], "game": game, "task_type": task_type})

    with open(data_dir / "manifest.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    counts = {s: sum(r["split"] == s for r in rows) for s in SPLITS}
    print(f"Packed {len(rows)} games into {data_dir}: {counts}; skipped {skipped}")


def extract_games(data_dir: str, dest: str) -> str:
    """Extract games.tar into `dest` (node-local disk) once per job; returns `dest`.

    Safe to call from several processes at once (one per GPU): the first extracts, the others wait.
    """
    Path(dest).mkdir(parents=True, exist_ok=True)
    marker = Path(dest) / ".extracted"
    with open(Path(dest) / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not marker.exists():
            with tarfile.open(Path(data_dir) / "games.tar") as tar:
                tar.extractall(dest, filter="data")
            marker.touch()
    return dest


def load_manifest(data_dir: str, split: str) -> list[dict[str, str]]:
    with open(Path(data_dir) / "manifest.jsonl") as f:
        return [r for r in map(json.loads, f) if r["split"] == split]


if __name__ == "__main__":
    build_archive(Path(sys.argv[1]), Path(sys.argv[2]))
