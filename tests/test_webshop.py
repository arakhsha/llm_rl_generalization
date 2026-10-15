import os
import re
from pathlib import Path

import pytest

from llm_rl.webshop import WebshopEnv, format_page, normalize_command

WEBSHOP_DIR = os.environ.get("WEBSHOP_DIR", "")
needs_webshop = pytest.mark.skipif(
    not (Path(WEBSHOP_DIR) / "search_engine/indexes/.done").exists(),
    reason="WebShop not set up yet (setup_webshop.sbatch)",
)
NAVIGATION = {"back to search", "next >", "< prev", "description", "features", "reviews", "buy now"}


def test_normalize_command():
    assert normalize_command("  Click[Buy  Now]. ") == "click[buy now]"
    assert normalize_command('"search[red shoes]"') == "search[red shoes]"


def test_format_page():
    text = "WebShop [SEP] Instruction: [SEP] i need red shoes [SEP] Search"
    assert format_page(text) == "Search"
    text = "Instruction: [SEP] i need red shoes [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next >"
    assert format_page(text) == "Back to Search\nPage 1 (Total results: 50)\nNext >"


def available(obs: str) -> list[str]:
    return re.findall(r"(?:click|search)\[[^\]]*\]", obs.rsplit("Available actions: ", 1)[1])


@pytest.fixture(scope="module")
def env():
    return WebshopEnv(WEBSHOP_DIR, max_steps=6)


@needs_webshop
def test_search_click_buy(env):
    obs = env.reset(goal=0)
    assert obs.startswith("Instruction: ") and available(obs) == ["search[<query>]"]
    instruction = obs.split("\n", 1)[0].removeprefix("Instruction: ")

    obs = env.act("click[buy now]")  # not on this page
    assert obs.startswith("Invalid action") and env.stats()["invalid"] == 1

    obs = env.act(f"search[{instruction}]")
    assert "Page 1 (Total results:" in obs and "click[next >]" in available(obs)
    products = [a for a in available(obs) if a.startswith("click[") and a[6:-1] not in NAVIGATION]
    assert len(products) == 10, obs
    print(obs)

    obs = env.act(products[0])
    assert "click[buy now]" in available(obs) and "search[<query>]" not in available(obs)
    print(obs)

    obs = env.act("click[buy now]")
    stats = env.stats()
    assert obs.endswith("The episode is over.") and stats["done"] and stats["bought"]
    assert 0.0 <= stats["reward"] <= 1.0 and stats["won"] == (stats["reward"] == 1.0)
    assert set(stats["reward_info"]) >= {"r_type", "r_att"}
    assert stats["steps"] == 4 and stats["invalid"] == 1
    assert env.act("click[back to search]").startswith("The episode is over")


@needs_webshop
def test_search_from_any_page(env):
    strict = WebshopEnv(WEBSHOP_DIR, max_steps=6, search_any_page=False)  # shares the loaded server
    for e in (env, strict):
        e.reset(goal=3)
        obs = e.act("search[shoes]")
        product = next(a for a in available(obs) if a.startswith("click[") and a[6:-1] not in NAVIGATION)
        e.act(product)
    assert "Page 1 (Total results:" in env.act("search[shirt]")
    assert strict.act("search[shirt]").startswith("Invalid action")


@needs_webshop
def test_step_limit_and_goals(env):
    first = env.reset(goal=1)
    for _ in range(5):
        assert not env.act("jump").endswith("The episode is over.")
    assert env.act("jump").endswith("Out of actions. The episode is over.")
    assert env.stats() == {
        "won": False, "done": True, "steps": 6, "invalid": 6, "reward": 0.0, "bought": False, "reward_info": None,
    }
    # Replaying a goal gives the same instruction, and the goals differ.
    assert env.reset(goal=1) == first and env.reset(goal=2) != first
