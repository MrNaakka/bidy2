1. **High — Agent can forge verifier rewards.** [scripts/build_tasks.py:210](/home/user/bidy2/scripts/build_tasks.py:210)  
   The verifier runs in the shared agent container, whose Dockerfile defaults to root, and trusts mutable executables such as `grep`. An agent can install `/usr/local/bin/grep` that always exits successfully, then stop without submitting. Both correctness checks pass and the verifier writes reward 1. Killing background processes does not remove modified executables. I verified these exact predicates accept genuine zero-reward JSON with a replacement `grep`.  
   **Fix:** Run verification in a trusted container, or restrict the agent to an unprivileged account that cannot modify verifier executables, dependencies, or output files.

2. **High — Public eval seeds allow perfect answers without viewing pixels.** [framehunt/splits.py:10](/home/user/bidy2/framehunt/splits.py:10), [framehunt/server.py:250](/home/user/bidy2/framehunt/server.py:250)  
   The committed seeds determine every answer, and free `info()` metadata identifies the item. All 30 eval items have distinct `n_frames`. A policy can precompute `{n_frames: answer}` from the published generators, then call only `info()` and `submit()`. I reproduced **30/30 correct with zero pixels spent**. This bypasses the intended sampling task entirely.  
   **Fix:** Evaluate on a private seed manifest with sufficient entropy, shared between reference and agent runs. Preserve that manifest for reproducibility without exposing it to the policy.

3. **High — The generated reference cannot demonstrate a positive gap.** [scripts/build_tasks.py:225](/home/user/bidy2/scripts/build_tasks.py:225), [scripts/mcp_smoke.py:75](/home/user/bidy2/scripts/mcp_smoke.py:75)  
   Every generated `solution/solve.sh` invokes the smoke client, which submits `PLUMBING-CHECK`. That answer cannot match any scene’s answer format, so the reference always scores zero. I checked its view/submit sequence against all three scene kinds; each returned reward 0. Frontier scores cannot fall below this reference.  
   **Fix:** Replace the placeholder with the promised MCP-only reference policy and demonstrate its scores on the evaluation split.

4. **Medium — The required reproducible evaluation entry point is absent.** [README.md:1](/home/user/bidy2/README.md:1)  
   The README contains only `# bidy2`; HEAD has no `scripts/eval.sh` or implementations of the documented baselines. Judges cannot run one command to reproduce the reference, three agent runs, and both baselines. The task builder only generates directories.  
   **Fix:** Implement that evaluation runner and document its invocation, including fixed seeds and retained run logs.

Validation: 182 tests passed, 5 deselected. `uv run` was blocked by the read-only cache; the broader run stalled in live MCP testing. Docker socket access was denied. No files were edited.

Verdict: Not ready for judging: reward integrity is bypassable, and the reference comparison and reproducible evaluation are incomplete.