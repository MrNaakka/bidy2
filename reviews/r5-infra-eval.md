1. **High — Cleanup forgets unresolved creates.** [infra/verda/verda_cli.py:551](/home/user/bidy2/infra/verda/verda_cli.py:551)  
   `delete_by_hostnames` clears the intent even when no matching instance was found. If a create request times out while Verda is still processing it, cleanup can successfully list nothing and erase the intent. The instance subsequently appears, bills indefinitely, and reconcile only warns that it is unknown. Reproduced with mocks. **Fix:** Retain unmatched intents for later reconciliation; clear them only after confirmed deletion or definitive rejection.

2. **High — Concurrent reconcile can erase creation tracking.** [infra/verda/verda_cli.py:646](/home/user/bidy2/infra/verda/verda_cli.py:646), [infra/verda/verda_cli.py:1469](/home/user/bidy2/infra/verda/verda_cli.py:1469)  
   Other cloud commands reconcile outside the watchdog lock and save their entire earlier state snapshot. Running `instances` during provisioning can overwrite a newly persisted creation intent. If the creator then dies, the paid instance becomes untracked. An in-memory interleaving reproduced the lost intent. **Fix:** Serialize state transactions across every command, including reconciliation, and avoid saving stale snapshots.

3. **Medium — Kept downloads still accept stale completion for different requests.** [infra/verda/verda_cli.py:735](/home/user/bidy2/infra/verda/verda_cli.py:735), [infra/verda/verda_cli.py:801](/home/user/bidy2/infra/verda/verda_cli.py:801)  
   Run `download --repos Qwen/Qwen3.8-27B-FP8 --keep`, then `download` requesting both checkpoints. The second invocation reuses the old attempt, accepts its existing `done`, and records BF16 weights as available although they were never downloaded. Reproduced with mocks; subsequent BF16 provisioning fails. **Fix:** Persist the requested repositories and revisions with each attempt; reuse only matching jobs and record only what that job verified.

4. **Medium — Qwen-only evaluation fails its credential check.** [scripts/eval.sh:61](/home/user/bidy2/scripts/eval.sh:61)  
   Both Qwen modes still require `ANTHROPIC_API_KEY` before reaching model selection. A setup containing the documented Qwen credentials exits with status 2. **Fix:** Validate the credentials required by the selected model.

5. **Medium — Retries discard failures after pixel use.** [scripts/eval.sh:56](/home/user/bidy2/scripts/eval.sh:56)  
   Harbor’s default retry policy includes `NonZeroAgentExitCodeError`. For example, Qwen can inspect frames, exceed vLLM’s image limit, and receive a fresh attempt with a fresh pixel budget. Harbor deletes the failed attempt’s directory and retains the replacement result; a mocked Harbor queue confirmed this. The replacement retains no earlier pixels or context, but discarding these failures still biases evaluation. **Fix:** Restrict retries to failures before agent execution and preserve attempt logs.

6. **Medium — Interrupted creation loses LoRA recovery configuration.** [infra/verda/verda_cli.py:424](/home/user/bidy2/infra/verda/verda_cli.py:424), [infra/verda/verda_cli.py:606](/home/user/bidy2/infra/verda/verda_cli.py:606)  
   Creation intents omit LoRA configuration, which is saved only after creation returns. Kill the CLI during a LoRA deployment, let reconcile adopt it, then preempt that instance: unattended recovery provisions only the base model because the adapter path was never persisted. **Fix:** Save the desired serving configuration before submitting creation and preserve it when adopting an orphan.

Validation: `bash -n` passed. The requested pytest command was blocked by the read-only filesystem; 11 existing test functions passed directly, alongside the in-memory probes above. No files were edited.

Verdict: Changes required before unattended operation or scored evaluation.