"""Text environments spread over worker processes, so a batch of episodes steps in parallel."""

import multiprocessing as mp
from collections import defaultdict


def _env_worker(conn, env_cls: type, env_kwargs: dict) -> None:
    """Serves batches of (slot, method, kwargs) requests on its own environment instances, in order."""
    if hasattr(env_cls, "init_worker"):  # per-process setup shared by the worker's environments
        env_cls.init_worker(**env_kwargs)
    envs = {}
    while (batch := conn.recv()) is not None:
        results = []
        for slot, method, kwargs in batch:
            if slot not in envs:
                envs[slot] = env_cls(**env_kwargs)
            env = envs[slot]
            try:
                results.append((getattr(env, method)(**kwargs), env.stats()))
            except Exception as e:  # reported to the caller, like a failing tool call
                results.append((e, env.stats()))
        conn.send(results)


class EnvPool:
    """Environments of one class spread over worker processes.

    A slot is one environment; slot i lives on worker i % num_workers, and a slot's requests within a
    batch run in order. Environments implement reset(**row) and act(command) returning an observation,
    and stats() returning a dict; an optional classmethod init_worker(**env_kwargs) runs once per worker
    at startup.

    Workers are forked, so create the pool before CUDA, vLLM or wandb start (spawned workers would
    re-import the `python -m` main module, and with it torch and vLLM, in every worker).
    """

    def __init__(self, env_cls: type, num_workers: int, **env_kwargs):
        ctx = mp.get_context("fork")
        self._conns = []
        for _ in range(num_workers):
            parent, child = ctx.Pipe()
            ctx.Process(target=_env_worker, args=(child, env_cls, env_kwargs), daemon=True).start()
            self._conns.append(parent)

    def run(self, requests: list[tuple[int, str, dict]]) -> list[tuple[str, dict]]:
        """Run (slot, "reset" | "act", kwargs) requests; returns (observation, stats) per request."""
        by_worker = defaultdict(list)
        for i, (slot, method, kwargs) in enumerate(requests):
            by_worker[slot % len(self._conns)].append((i, (slot, method, kwargs)))
        for w, items in by_worker.items():
            self._conns[w].send([req for _, req in items])
        results = [None] * len(requests)
        for w, items in by_worker.items():
            for (i, _), result in zip(items, self._conns[w].recv(), strict=True):
                results[i] = result
        return results

    def close(self) -> None:
        for conn in self._conns:
            conn.send(None)
