# AGENTS.md

## Context

This repository is our team's entry to AfterQuery's "Break the Agent" challenge at Junction 2026, a 48-hour hackathon. The full challenge text is in [docs/challenge.md](docs/challenge.md). Read it when you need details this file leaves out.

## Goal

Build a reinforcement learning environment where today's best AI agents fail in ways we can explain. The environment needs:

- A setting worth learning: a game, a simulation, a software task, a real-world process, or something else
- A clear goal, actions the agent can take, and observations it gets back
- A reward we can check, which only goes up when the agent actually makes progress

Then we run a frontier agent in it and compare it with a stronger reference run that we can show and reproduce. We explain where the agent falls short and why.

## Harness

AfterQuery provides the harness. Harbor runs the OpenCode agent, which talks to our environment over MCP. We also get a scoring skeleton and frontier model API credits. So our environment is an MCP server: its tools are the agent's actions, and their responses are its observations.

## How judges score it

| Criterion | Points | Full points when |
| --- | --- | --- |
| The gap | 30 | The frontier agent scores below our reference run every time, and both results reproduce across repeated runs. |
| Meaningful difficulty | 30 | We can explain the agent's main mistakes and tie them to decisions in the task. The agent has the information and tools it needs to do better. |
| Reward integrity | 25 | The reward holds up against exploits. Doing nothing or stopping early scores close to zero. Doing the task well is the best way to score. |
| Craft | 15 | The rules make sense for the setting. Judges can run the evaluation from the README with one command and get our reported results. |

The judges rerun everything with the standard harness: our reference run, three agent runs, a do-nothing baseline, and a stop-early baseline. They then spend 15 minutes trying to hack the reward. Each team also gets a 10-minute demo.

Tiebreaker: training a smaller model on experience from our environment and showing that it improves.

## Rules for building

- Difficulty must come from decisions the agent makes. Hidden information, trick wording, impossible timing, or broken tools score zero.
- Results must be reproducible. Fix seeds and log every run.
- Treat reward hacking as a test case. For every reward term, ask how an agent could raise it without doing the task.
- The README must have one command that runs the full evaluation.

## Starting points the organizers named

- Planning several steps ahead
- Exploring an unfamiliar system
- Using tools
- Managing limited resources
- Adapting when new information arrives

A small environment with a well-chosen challenge is enough.
