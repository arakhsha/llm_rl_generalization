"""WebShop (Yao et al., 2022) as a multi-turn tool environment, for evaluation.

The model gets a shopping instruction such as "i need a long clip-in hair extension which is natural
looking, and price lower than 40.00 dollars" and shops by calling the same `act` tool as in ALFWorld, with
WebShop's actions as commands: search[<query>] and click[<button>]. The episode ends when it clicks
"buy now". The score is WebShop's reward in [0, 1] for how well the product's type, attributes, options
and price match the instruction; an episode succeeds when the score is 1.

Runs WebShop's own text environment (web_agent_site, imported from the checkout in $WEBSHOP_DIR) on the
full product set (1.18M products) with the human-written instructions, as set up by
scripts/setup_webshop.sbatch. WebShop shuffles the goals with a fixed seed; goals 0-499 are the usual test set.
"""

import random
import re
import resource
import sys
import time
from statistics import mean

TEST_GOALS = (0, 500)  # goal indices [start, stop) of the test split

SYSTEM_PROMPT = """\
You are a shopping agent in WebShop, an online store. Buy the product that best matches the instruction \
in the first observation by calling the `act` tool with one action at a time, reading each observation \
before choosing the next action.

Actions:
- search[<query>]: search for products, from the search page
- click[<button>]: click a button, product ID or product option on the page, such as click[b07xyz1234], \
click[large] or click[buy now]

Each observation ends with the actions available on the page. Open a product to see its details and \
options, select the options (such as size and color) that the instruction asks for, then click[buy now]. \
The number of actions is limited, so be efficient."""

USER_PROMPT = "Here is your first observation.\n\n"


def build_prompt() -> list[dict[str, str]]:
    """Conversation prefix; the environment's reset() appends the first observation to the user turn."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": USER_PROMPT}]


def normalize_command(command: str) -> str:
    """Lower-case, collapse whitespace and drop surrounding quotes / a trailing period."""
    command = " ".join(command.strip().strip("\"'`").split()).lower()
    return command.rstrip(".")


def format_page(observation: str) -> str:
    """WebShop's text observation (page elements joined by [SEP]) as one element per line, without the
    instruction shown at the top of every page."""
    parts = [p.strip() for p in observation.split("[SEP]") if p.strip()]
    if "Instruction:" in parts:
        parts = parts[parts.index("Instruction:") + 2 :]
    return "\n".join(parts)


def goal_rows(start: int, stop: int) -> list[dict[str, int]]:
    """Episode rows (WebshopEnv.reset kwargs) for goals [start, stop)."""
    return [{"goal": i} for i in range(start, stop)]


class WebshopEnv:
    """One WebShop session at a time: reset() starts a session for a goal, act() runs an action.

    The environments of a worker process share one WebShop SimServer (products, goals and search
    engine), loaded once by init_worker. The signature and docstring of `act` define the tool schema
    shown to the model.
    """

    _server = None

    def __init__(self, webshop_dir: str, max_steps: int, seed: int = 0, search_any_page: bool = True):
        self._webshop_dir = webshop_dir
        self._max_steps = max_steps
        self._seed = seed
        self._search_any_page = search_any_page
        self._env = None
        self._actions = None  # WebShop's get_available_actions() for the current page
        self.reward = 0.0
        self.bought = self.done = False
        self.steps = self.invalid = 0
        self.reward_info = None

    @classmethod
    def init_worker(cls, webshop_dir: str, seed: int = 0, **kwargs) -> None:
        """Load WebShop's products, goals and search index; takes a few minutes for the full product set."""
        if cls._server is not None:
            return
        start = time.time()
        if webshop_dir not in sys.path:
            sys.path.insert(0, webshop_dir)
        from web_agent_site.envs.web_agent_text_env import SimServer
        from web_agent_site.utils import DEFAULT_ATTR_PATH, DEFAULT_FILE_PATH

        assert DEFAULT_FILE_PATH.endswith("items_shuffle.json") and DEFAULT_ATTR_PATH.endswith("items_ins_v2.json"), (
            "web_agent_site/utils.py must point to the full product files (see scripts/setup_webshop.sbatch)"
        )
        # Prices of products with a price range, and the price limits in the instructions, come from `random`.
        random.seed(seed)
        cls._server = SimServer("http://127.0.0.1:3000", DEFAULT_FILE_PATH, human_goals=True)
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB -> GB
        print(
            f"WebShop: {len(cls._server.all_products)} products, {len(cls._server.goals)} goals, "
            f"loaded in {time.time() - start:.0f}s, max RSS {max_rss:.1f} GB",
            flush=True,
        )

    def reset(self, goal: int, **kwargs) -> str:
        if self._env is None:
            self.init_worker(self._webshop_dir, self._seed)
            from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

            # Sessions are named <prefix><goal>, so environments in one worker playing the same goal stay apart.
            self._env = WebAgentTextEnv(observation_mode="text", server=self._server, session_prefix=f"{id(self)}_")
        obs, _ = self._env.reset(session=goal)
        self.reward = 0.0
        self.bought = self.done = False
        self.steps = self.invalid = 0
        self.reward_info = None
        instruction = self._server.user_sessions[self._env.session]["goal"]["instruction_text"]
        return f"Instruction: {instruction}\n\n{self._observation(format_page(obs))}"

    def act(self, command: str) -> str:
        """Run one action in the web shop.

        Args:
            command: The action to run, e.g. "search[red running shoes]" or "click[buy now]".

        Returns:
            The observation after the action.
        """
        if self.done:
            return "The episode is over. Do not call act again."
        command = normalize_command(command)
        self.steps += 1
        if self._is_valid(command):
            obs, reward, done, _ = self._env.step(command)
            if done:  # clicked "buy now"
                self.reward, self.bought, self.done = reward, True, True
                self.reward_info = self._server.user_sessions[self._env.session].get("verbose_info")
                return "You bought the product. The episode is over."
            feedback = format_page(obs)
        else:
            self.invalid += 1
            feedback = "Invalid action: use one of the available actions."
        if self.steps >= self._max_steps:
            self.done = True
            return f"{feedback}\n\nOut of actions. The episode is over."
        return self._observation(feedback)

    def stats(self) -> dict:
        return {
            "won": self.reward == 1.0,
            "done": self.done,
            "steps": self.steps,
            "invalid": self.invalid,
            "reward": self.reward,
            "bought": self.bought,
            "reward_info": self.reward_info,
        }

    def _is_valid(self, command: str) -> bool:
        """WebShop ignores clicks on buttons that are not on the page; they count as invalid here.

        WebShop runs search[...] from any page, although only the search page has a search bar (and lists
        search among its available actions); search_any_page=False only allows it on the search page.
        """
        m = re.fullmatch(r"(search|click)\[(.+)\]", command)
        if m is None:
            return False
        name, arg = m.groups()
        if name == "search":
            return (self._search_any_page or self._actions["has_search_bar"]) and bool(arg.strip())
        return arg in self._actions["clickables"] and arg != "search"

    def _observation(self, text: str) -> str:
        """`text` followed by the available actions, which are refreshed for the next step."""
        self._actions = self._env.get_available_actions()
        actions = ["search[<query>]"] if self._actions["has_search_bar"] else []
        # "search" is the search page's submit button, used through search[...].
        actions += [f"click[{c}]" for c in self._actions["clickables"] if c != "search"]
        return f"{text}\n\nAvailable actions: {', '.join(actions)}"


def webshop_metrics(records: list[dict]) -> dict[str, float]:
    """WebShop's score (mean reward x 100, 0 for episodes that bought nothing) and the purchase rate."""
    return {
        "score": 100 * mean(r["reward"] for r in records),
        "bought": mean(r["bought"] for r in records),
    }
