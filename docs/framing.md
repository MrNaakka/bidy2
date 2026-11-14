# FrameHunt: pitch framing

Notes for the deck and the demo video. Numbers and charts for slides are in
[results/tables.md](results/tables.md).

## The idea

**Every agent pays to look.** Reading a video, a screenshot, a log or a codebase costs
tokens, and everything the agent looks at stays in its context. Careless looking makes an
agent slower, more expensive and less focused.

So looking efficiently is a core agent skill, and few training setups target it.

Tagline: **"Good agents don't see more. They look smarter."**

## Errors and wasted budget go together

In our environment, the failures and the wasted budget show up together:

- Opus ends its wrong answers with the budget almost gone (mean 117k of 120k). It spends
  heavily early, then has nothing left to check its answer.
- The reference uses less than half the budget (50k mean) and scores 40/40.
- Even when Opus is right, it spends about twice as much (106k).
- A 27B model fine-tuned on trajectories that always verify before reading went from
  0.25 to 0.62. It used about two thirds of the base model's pixels and beat Opus on
  logstream (0.90 vs 0.43).

Line for the slide: **"Efficient looking is accurate looking."**

## What the environment does

The agent gets a question about a long video. It can't watch the video. It can only ask
for single frames, and every pixel comes out of a fixed budget.

- **Reward:** 1 for the right answer, else 0.
- **Efficiency:** a hard budget, not a reward term. If efficiency earned points, doing
  nothing would score. This way the only way to score is to answer correctly.

## The skit: the pizza agent

Hackathon, 3am. The team tells an AI agent: "Order 5 pizzas."

1. The agent clicks **Pay**.
2. A toast flashes for 1 second: **"Payment successful ✓"**. The agent's next screenshot
   comes 2 seconds later. The toast is gone.
3. Agent: *"I don't see a confirmation. Retrying."*
4. Again. And again. Phone: *"Your card was charged €62.50"* ×4.
5. Doorbell. 20 pizzas.

What the skit shows:

- **Looking too rarely** misses the confirmation and buys 20 pizzas.
- **Looking constantly** floods the context and burns the credits.
- **The right answer** is knowing when and where to look. That's what we train.

**Bridge slide:** *"The confirmation was on screen. The agent just didn't look at the
right moment. That's what our environment tests."*

## Keep in mind

- The skit is only the hook. Don't present invented lines as Opus's real words.
- "A cluttered context makes agents worse" is the motivation, not something we measured.
  What we measured is that failures come with an exhausted budget.
- Toasts is our control: same mechanism as logstream, but the check is cheap and right
  next to the answer. Opus scores 0.80 there and 0.43 on logstream. That fits "the cost of
  verifying drives the failures", but it is not proof: layouts and distractors differ too.
