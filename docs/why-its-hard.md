# Why it's hard

This page explains where Opus 5.5 loses points on FrameHunt and why. Numbers come from
its three EVAL runs (3 x 30 items, 90 trials, 36 failures). We checked every failure
against the regenerated ground truth and read the agent's final message.

## The agent has what it needs

- The instruction states the question, the media facts, the budget and the cost of
  every call.
- The trace (a counter, a badge, page numbers and headers) is visible on every frame at
  low resolution.
- Every contact sheet comes with a text list that maps each tile to its frame.
- Every trap is visible on screen.

The reference script knows only the screen layout and the fonts. It scores 30/30 and
uses less than half the budget.

## The three kinds and their traps

| Kind | Setting | Question | Trap |
| --- | --- | --- | --- |
| `logstream` | terminal recording of a failing service, 3-5 new log lines per second | request id of the K-th ERROR line | One frame early, the (K-1)-th ERROR line is the newest. A few frames late, newer lines have pushed it up. Checking the `errors: N` counter on the exact frame costs a few hundred pixels. |
| `dialogs` | desktop app; an error dialog pops up and closes (the common E2E-video case) | code shown in the K-th error dialog | The code is small. The dialog stays up for a while, so timing is forgiving. |
| `scandoc` (document) | scanned annual report, 150-400 page images, frame index = PDF page | one amount in one note's table at a given year-end | Printed page numbers are not PDF indexes (roman front matter, unnumbered inserted sheets). The comparative-year table with the same caption is on the facing page. A full page at 1/4 costs more than the budget. |

## What has to go right

Each item is a short chain of decisions:

1. Plan the budget.
2. Find the exact moment on the cheap trace.
3. **Verify it.**
4. Decide where to crop.
5. Spend on one sharp crop.

Most failures break step 3 or step 4.

## logstream: 17 of 30 trials failed

**Reading the log before the line exists (10 of 17).** Every sharp crop of the log was
taken before the K-th ERROR line was printed. Five were only 1 to 3 frames (0.1-0.3 s)
early. At that frame the `errors:` counter still shows K-1. In 9 of the 10, the agent had
already bought a contact-sheet tile of that exact frame showing K-1 (tiles 10-32 px tall)
and misread it.

A direct look at the counter on the frame it was about to read costs a few hundred
pixels. Opus did not take that look in any of the 30 logstream trials.

**Right moment, wrong row (5 of 17).** The agent read at or after the event, but the
target row was outside every sharp crop. New lines arrive 3-5 per second, so "the newest
line" moves. In 4 of the 5 it submitted another line's well-formed id (DEBUG, WARN,
INFO); in 1 it guessed. The red ERROR label is visible at low resolution, so finding the
row first is cheap.

**No budget left to recover.** All 17 failures ended with under 5k pixels left (median
330). The first sharp look at the log came after a median of 88k pixels. The reference
needs about 51k for the whole item.

## scandoc: 15 of 30 trials failed

| Cause | Failures |
| --- | --- |
| Wrong page: assumed printed page = PDF page +/- a constant, missing the roman front matter or an inserted sheet | 5 |
| Read the facing comparative-year table (2 of them gave the exact other-year value of the right cell) | 3 |
| Right table, wrong row | 2 |
| Digits misread at 1/3 scale | 2 |
| Budget gone before reaching the table, then a guess | 3 |

14 of the 15 ended with under 5k pixels left.

## dialogs: 4 of 30 trials failed

- 2 looked before the dialog opened or after it closed.
- 2 had crops that missed the code field, then read the code at 1/3 scale.

## Across all kinds

- 32 of 36 failures end with a final message saying the agent is guessing or unsure.
  For correct answers it is 16 of 54.
- 34 of 36 failures end with under 5k pixels left. By the time the agent says it is
  unsure, there is no budget left to check.
- Even when right, Opus spends about twice what the reference does (109k vs 53k per
  item).
- The 4 failures that don't hedge are mostly the traps working as designed: the facing
  page, a wrong row in a table, a wrong log row, and one misread amount.

Per-kind scores and charts: [results/tables.md](results/tables.md). Item rules and
generator guarantees: [design.md](design.md).
