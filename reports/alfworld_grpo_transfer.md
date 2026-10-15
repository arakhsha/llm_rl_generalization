# ALFWorld GRPO training: effect on ALFWorld and WebShop

**Date:** 2026-10-15

**Model compared:** Qwen3.5-2B (base) vs. the same model after GRPO + LoRA training on ALFWorld (run
`alfworld_grpo_20261014-190342`).

## Summary

- **ALFWorld (the training environment):** success rose from about 2% to about 80% on both validation splits. Rooms
  the model never saw in training improved as much as seen rooms (1.5% → 78.4% vs 2.9% → 80.0%). Invalid commands per
  game fell 3–4×, from about 22–27 to 6–8. As a share of all commands, however, they fell only from 46–57% to 34–38%.
- **WebShop (not trained on):** there is no transfer gain.
  - Success, a perfectly matching purchase, is unchanged: 22.0% → 24.4%, not significant (p = 0.31).
  - The benchmark score fell by 9.9 points (53.8 → 43.9).
  - Invalid actions rose: from 0.7 to 1.9 per episode, or 13.5% → 19.6% of actions.
- **Why the WebShop score fell:** the tuned model brought ALFWorld's exploration habits to WebShop. It searched and
  opened more pages, often lost track of which page it was on, and ran out of its 15 actions before buying in 37% of
  episodes (base: 12%). When it did buy, its purchases matched better than the base model's.

## Setup

**Training (ALFWorld)**

| Setting | Value |
|---|---|
| Method | GRPO with LoRA (rank 32, α 64) on all language-model linear layers |
| Thinking | Off |
| Length | 150 steps of 64 episodes each (8 per game), sampling temperature 1.0 |
| Data | ALFWorld's training games |
| Reward | 1 if the game is won, otherwise 0; nothing else is rewarded or penalised |
| Interface | The model calls an `act` tool with one text command per turn |
| Commands shown | Each observation lists the commands available at the current location, except `go to`, `look` and `inventory` |
| Limits | 50 commands and 6,144 tokens per game |

**ALFWorld evaluation**

| Setting | Value |
|---|---|
| Splits | `valid_seen` (140 games, rooms seen in training) and `valid_unseen` (134 games, new rooms) |
| Decoding | Greedy |
| Limits | Same as training: 50 commands, 6,144 tokens |
| Tuned model numbers | The step-150 validation inside the training run |
| Base model numbers | A separate run with the same settings |

**WebShop evaluation**

| Setting | Value |
|---|---|
| Test set | The 500 human-written test instructions |
| Product set | The full set of 1.18M products |
| Decoding | Greedy |
| Limits | 15 actions and 16,384 tokens per episode |
| Interface | The same `act` tool, taking WebShop's `search[...]` and `click[...]` actions |
| Observations | Each page is shown with its list of available actions |
| Search rule | `search[...]` is accepted from any page, as in WebShop's own environment |
| Identical goals | Both models received the same instructions and price limits; the base model's first action matched on 500/500 goals across two runs |

**Metric definitions**

| Metric | ALFWorld | WebShop |
|---|---|---|
| **Reward** (main metric) | Success rate: share of games won | **Score:** 100 × average reward. The reward (0–1) rates how well the purchased product matches the instruction: type, attributes, options and price. It is 0 if nothing was bought. |
| | | **Success:** share of episodes with reward exactly 1 |
| **Invalid action** | A command the game answers with "Nothing happens.": unparseable, or impossible in the current state | An action not available on the current page (for example, `click[buy now]` on a page that has no such button) |
| **Invalid rate** | Invalid actions ÷ all actions, pooled over all episodes | Same |

Each invalid action still uses up one of the episode's steps.

## ALFWorld (training environment)

### Reward

| | Base, seen | Tuned, seen | Base, unseen | Tuned, unseen |
|---|---|---|---|---|
| **Success** | **2.9%** (4/140) | **80.0%** (112/140) | **1.5%** (2/134) | **78.4%** (105/134) |
| Ran out of commands (50) | 82.9% | 5.7% | 92.5% | 9.0% |
| Ran out of tokens (out of tokens or truncated) | 14.3% | 14.3% | 5.2% | 12.7% |
| Stopped calling the tool | 0% | 0% | 0.7% | 0% |
| Average commands per game | 47.3 | 17.4 | 48.7 | 20.4 |
| Average tokens per game | 4,561 | 2,477 | 4,301 | 2,803 |

The base model almost never finishes a game: it uses up all 50 commands. The tuned model wins 4 in 5 games, in about a
third as many commands. The gap between seen and unseen rooms is small (80.0% vs 78.4%). So the policy generalises to
new room layouts rather than memorising the training rooms.

Success by task type:

| Task | Base, seen | Tuned, seen | Base, unseen | Tuned, unseen |
|---|---|---|---|---|
| pick_and_place_simple | 11% | 97% (34/35) | 8% | 88% (21/24) |
| pick_heat_then_place_in_recep | 0% | 88% (14/16) | 0% | 87% (20/23) |
| pick_clean_then_place_in_recep | 0% | 85% (23/27) | 0% | 77% (24/31) |
| pick_two_obj_and_place | 0% | 79% (19/24) | 0% | 76% (13/17) |
| pick_cool_then_place_in_recep | 0% | 68% (17/25) | 0% | 90% (19/21) |
| look_at_obj_in_light | 0% | 38% (5/13) | 0% | 44% (8/18) |

Every task type went from (near) zero to most games won, except `look_at_obj_in_light`, which remains the weak spot.
Samples are small: one game changes a cell by 3–8 points.

Progress during training (greedy validation success, seen / unseen):

| Step | 0 (base) | 25 | 50 | 75 | 100 | 125 | 150 |
|---|---|---|---|---|---|---|---|
| Success | 2.9 / 1.5 | 32.9 / 18.7 | 44.3 / 42.5 | 48.6 / 50.7 | 37.9 / 33.6 | 63.6 / 62.7 | **80.0 / 78.4** |
| Invalid rate | 56.9 / 46.0 | 38.8 / 50.5 | 17.1 / 23.0 | 16.4 / 20.6 | 21.8 / 31.1 | 47.6 / 48.2 | 34.3 / 38.0 |

Success was still rising at the last step, so longer training would probably help.

### Validity of actions

| | Base, seen | Tuned, seen | Base, unseen | Tuned, unseen |
|---|---|---|---|---|
| Invalid rate (share of commands) | 56.9% | 34.3% | 46.0% | 38.0% |
| Invalid commands per game | 26.9 | 6.0 | 22.4 | 7.7 |
| Valid commands per game | 20.4 | 11.4 | 26.3 | 12.7 |

- **Per game, invalid commands fell 3–4×.** This happens mostly because the tuned model finishes games in fewer
  commands.
- **As a share of commands, invalid commands fell less**, and not steadily during training. The invalid rate dropped
  to 16–23% by steps 50–75. It rose again to 48% at step 125, at the same time as the largest jump in success, and
  ended at 34–38%.
- **This is consistent with how the reward is defined.** An invalid command costs nothing except a step and some
  tokens, and the reward counts only wins. So the policy learned to finish games, not to respect the interface. A
  third of its commands still get "Nothing happens.", for example trying an action and moving on when it fails. A
  step or invalid-action penalty would be the lever if interface discipline matters.

## WebShop (not trained on)

### Reward

| | Base | ALFWorld-tuned | Paired difference (tuned − base) |
|---|---|---|---|
| **Score** | **53.8** | **43.9** | −9.9 (95% CI −14.1 to −5.7); tuned better on 115 goals, worse on 188 |
| **Success** (reward = 1) | **22.0%** | **24.4%** | +2.4 points; 65 goals only tuned wins, 53 only base wins; McNemar p = 0.31 |
| Bought something | 85.0% | 62.0% | |
| Average reward of purchases × 100 | 63.3 (425 purchases) | 70.8 (310 purchases) | |
| Ran out of actions (15) | 11.8% | 37.2% | |
| Stopped calling the tool | 2.8% | 0% | |
| Ran out of tokens | 0.4% | 0.8% | |
| Average actions per episode | 4.8 | 9.5 | |
| Average tokens per episode | 1,779 | 3,981 | |

ALFWorld training did not transfer as a gain in reward:

- **Success:** statistically unchanged.
- **Score:** significantly lower. The tuned model buys in far fewer episodes, and an episode without a purchase
  scores 0.
- **Purchase quality:** better when the tuned model does buy (70.8 vs 63.3).

### Validity of actions

| | Base | ALFWorld-tuned |
|---|---|---|
| Invalid rate (share of actions) | 13.5% | 19.6% |
| Invalid actions per episode | 0.66 | 1.86 |
| Invalid `click[buy now]` from a product's Description or Features page (no such button there) | 9 | 178 |

On WebShop, the interface got worse after ALFWorld training: invalid actions rose both as a share of actions and per
episode (2.8×).

- **Most of the increase comes from a new failure mode.** The tuned model opens a product's Description or Features
  page and then tries to buy from there, as if it were still on the product page.
- **The base model rarely does this,** because it rarely visits those pages.

### What changed in behaviour

| | Base | ALFWorld-tuned |
|---|---|---|
| Searches (all 500 episodes) | ~555 | ~1,359 |
| Product pages opened | 763 | 1,403 |
| Description and Features pages viewed | ~25 | 299 Features, 273 Description |

**Base model:** usually searches, clicks the first result, and buys. Its purchases are mediocre matches, but it almost
always buys. Its failures are mostly repetition loops, such as clicking the same link 14 times.

**ALFWorld-tuned model:** behaves like an ALFWorld agent.

- **It explores more:** it searches repeatedly and inspects several products and their detail pages before
  committing.
- **This helps purchase quality:** its purchases match better, and it wins slightly more often.
- **But it hurts within a 15-action budget:**
  - It loses track of which page it is on, which produces the invalid buy attempts.
  - It often runs out of actions before buying.
- **A plausible cause is how ALFWorld episodes end.** They end only with a win or at the step limit, so the model
  never needed to commit to a final, irreversible action, which WebShop requires.

### Sensitivity to the search rule

In WebShop's own environment, `search[...]` works from any page, although only the search page lists it. The main
results above use that rule. A stricter variant, where searching is only valid on the search page, hurts the tuned
model much more, because it often searches from other pages:

| Strict rule | Base | ALFWorld-tuned |
|---|---|---|
| Score | 53.5 | 38.1 |
| Success | 22.2% | 21.0% |
| Invalid rate | 14.0% | 44.1% |
| Ran out of actions | 11.8% | 46.0% |

The base model is essentially unaffected by the rule.

## Conclusions

1. **In-domain:** GRPO with a win-only reward turned a model that almost never completes ALFWorld (about 2%) into one
   that wins about 80%, including in unseen rooms. It had not plateaued after 150 steps.
2. **Interface validity in-domain:** invalid commands fell sharply per game, but only modestly as a share of commands.
   Because the reward ignores invalid actions, about a third of commands remain invalid.
3. **Out-of-domain (WebShop):** no transfer gain.
   - Success is unchanged.
   - The benchmark score drops by about 10 points.
   - Interface validity gets worse (2.8× more invalid actions per episode).
4. **The skills that transfer are partial.** The tuned model researches products more carefully, and its purchases
   match better when it buys. But it doesn't adapt its exploration to WebShop's tighter action budget, or to the need
   to finish by buying.

## Caveats

- **One training run and one seed.** The ALFWorld curve is noisy, for example the dip at step 100. The step-150
  numbers come from a single checkpoint.
- **Small ALFWorld per-task samples** (13–35 games per cell).
- **WebShop product data** comes from a third-party Hugging Face mirror (`YWZBrandon/webshop-data`, pinned revision),
  because WebShop's original Google Drive links are gone. It matches the paper: 1,181,436 products and 12,087
  instructions. Its small files are byte-identical to a second mirror.
- **WebShop setup choices** that affect absolute numbers:
  - the 15-action limit (as in verl-agent; longer limits would likely help the tuned model more)
  - newline-formatted pages with the available actions listed
  - thinking turned off
  - greedy decoding
- **The ALFWorld invalid rate is not comparable to WebShop's.** In ALFWorld, "Nothing happens." also covers commands
  that are well formed but impossible in the current state.

## Where the data is

| What | Location |
|---|---|
| Tuned model (adapter, merged model, validation history) | `/project/aip-sologen/aminr/llm_rl_generalization/models/alfworld_grpo_20261014-190342/` |
| Base ALFWorld evaluation | `/project/aip-sologen/aminr/llm_rl_generalization/eval_alfworld/20261014-182329/` |
| WebShop evaluations, base / tuned | `/project/aip-sologen/aminr/llm_rl_generalization/eval_webshop/20261015-071741-1758060/` and `…-1758061/` |
| WebShop evaluations, strict rule, base / tuned | `…/20261015-071603-1758023/` and `…/20261015-070401-1757749/` |
| wandb (`llm_rl_generalization` project) | `eval_webshop/base`, `eval_webshop/alfworld_grpo_20261014-190342`, and the training run `alfworld_grpo_20261014-190342` |

The evaluation directories hold per-episode transcripts. The training run directory (checkpoints, logs) is in
`/project/aip-sologen/aminr/llm_rl_generalization/runs/alfworld_grpo_20261014-190342/`.
