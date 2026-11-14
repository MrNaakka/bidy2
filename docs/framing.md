## The one-sentence version

**AI can't afford to watch a whole video, so it has to skim. We test whether it can skim smart, and Opus can't.**

## Why everyone gets it

Everyone has scrubbed through a long video to find one moment. A person does it the same way every time:

1. **Skim fast** with the scrub bar and blurry thumbnails to find roughly *when* it happens.
2. **Narrow down** to the exact moment.
3. **Pause and zoom** to read the detail.

That's cheap and it works. Our environment gives the AI exactly this job, with a limited "looking budget." Every pixel it looks at costs something, like vision tokens cost money in real life.

## What Opus does wrong (in the same plain terms)

- It **zooms in too early**, before knowing *when* to look, and burns half its budget on the first look.
- It finds roughly the right moment but **doesn't double-check it**, so it zooms into a frame that's slightly too early, before the thing it's looking for appears on screen.
- It **runs out of budget and guesses**, and admits it's guessing.

The smart version, our reference script and the model we fine-tune, uses less than half the budget and gets 30/30.

## The tagline options

- "Skimming is a skill. Frontier AI doesn't have it."
- "It's not that the AI can't see. It doesn't know *where* to look."
- "Like watching someone scrub a video by zooming into random frames."

My pick is the second. It's simple, it explains the failure, and it heads off the judges' "isn't this just bad vision?" objection.
