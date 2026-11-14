# FrameHunt: framing

## The big idea

**Every agent pays to look.** Reading a video, a screenshot, a log or a codebase costs tokens, and everything it looks at piles up in its context. Careless looking makes an agent slower, more expensive and less focused on the task.

So **looking efficiently is a core agent skill**, and almost nobody trains for it.

Tagline: **"Good agents don't see more. They look smarter."**

## Errors and efficiency are the same problem

In our environment, wasting budget is what causes the errors:

- Opus ends nearly every item with the budget almost gone (mean 117k of 120k on wrong answers). It spends heavily early, then has nothing left to check its answer.
- The reference uses less than half the budget (50k mean) and scores 40/40.
- Even when Opus is right, it spends about 2× as much (106k).
- A 27B model fine-tuned on our environment's trajectories learned to verify before reading: 0.25 → 0.62, with about two thirds of the base model's pixels, and beats Opus on logstream (0.90 vs 0.43).

**Efficient looking is accurate looking.**

## What our environment does

The agent gets a question about a long video. It can't watch the video. It can only ask for single frames, and every pixel costs from a fixed budget.

- **Reward:** 1 for the right answer, else 0.
- **Efficiency:** a hard budget, not a reward term. If efficiency gave points, doing nothing would score. This way, the only way to score is to answer correctly, and the only way to answer correctly is to look efficiently.

## The skit: the pizza agent

Hackathon, 3am. The team tells an AI agent: "Order 5 pizzas."

1. The agent clicks **Pay**.
2. A toast flashes for 1 second: **"Payment successful ✓"**. The agent's next screenshot is 2 seconds later. The toast is gone.
3. Agent: *"I don't see a confirmation. Retrying."*
4. Again. And again. Phone: *"Your card was charged €62.50"* ×4.
5. Doorbell. 20 pizzas.

The skit shows the trade-off:
- **Looks too rarely:** misses the confirmation, buys 20 pizzas.
- **Looks constantly:** floods the context, burns the credits.
- **Right answer:** know when and where to look. That's what we train.

**Bridge slide:** *"The confirmation was on screen. The agent just didn't look at the right moment. That's what our environment tests."*

## Keep in mind

- The skit is just the hook. Don't present invented lines as Opus's real words.
- "A cluttered context makes agents worse" is the vision, not something we measured. Our evidence is that wasting the budget leads to errors.
- Toasts is our control: same mechanism as logstream, but the check is cheap and right next to the answer. Opus scores 0.80 there vs 0.43 on logstream. It fits "verification cost drives the failures" but is not proof (layouts and distractors differ too).
- Numbers and charts for slides: [docs/results/](results/tables.md).
