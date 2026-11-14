1. **High — Reconcile deletes unrelated instances based only on their hostname.** [infra/verda/verda_cli.py:672](/home/user/bidy2/infra/verda/verda_cli.py:672)  
   Any untracked `fh-` instance is treated as ours and potentially deleted, without a creation intent or ledger record. **Scenario:** a manually created `fh-training` box is permanently deleted when someone runs `instances` or the watchdog. Reproduced with mocks. **Fix:** Require recorded ownership or a dedicated deployment identifier; report unknown instances without deleting them.

2. **High — The prefix change makes `serve-down` leave existing paid instances running.** [infra/verda/verda_cli.py:1221](/home/user/bidy2/infra/verda/verda_cli.py:1221)  
   Previous versions created `fh-qwen-serve…` instances; lifecycle commands now select only `fh-serve-…`. Reconcile preserves a tracked legacy instance, so shutdown never selects it. **Scenario:** update the checkout while an old serving instance exists, then run `serve-down`: it succeeds and clears the tunnel URL while the instance continues billing. Reproduced with mocks. **Fix:** Select tracked serving IDs regardless of hostname and support legacy prefixes during migration.

Validation: `bash -n` passed. The requested pytest command was blocked by the read-only filesystem; 15 existing file-free tests passed directly. Harbor’s current `AgentSetupTimeoutError` allowlist permits retries only before agent execution, so it does not grant another attempt after pixel use.

Verdict: Changes required before unattended operation.