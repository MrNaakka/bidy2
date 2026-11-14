1. **High — Running jobs never enforce the budget.** [verda_cli.py:1694](/home/user/bidy2/infra/verda/verda_cli.py:1694)  
   `max_eur` is checked only during provisioning. A slow or hung active trainer continues billing after exhaustion. A mocked tick with €100 spent against a €40 cap returned `running` without checking spend. **Fix:** Enforce the cap on every tick, stop and delete the box when exhausted, and persist that training is disabled.

2. **High — Failed starts bypass the restart limit.** [verda_cli.py:1749](/home/user/bidy2/infra/verda/verda_cli.py:1749)  
   Upload/start errors escape before `started=True`; restart counting requires that flag. Repeated `systemctl start` failures therefore leave a ready GPU billing indefinitely, including past the setup grace period. **Fix:** Count unsuccessful startup attempts, bound retries, and invoke cleanup when exhausted.

3. **High — Hash parsing breaks recovery and leaves the replacement GPU running.** [verda_cli.py:1558](/home/user/bidy2/infra/verda/verda_cli.py:1558)  
   Stamped files end with a newline, and the intervening `echo` adds another. Output is `DATA\n\nCODE\n`, so the parser reads an empty code hash. After a local trainer edit and preemption, it refuses recovery despite the correct code remaining on the volume; subsequent ticks repeat the refusal without cleanup. **Fix:** Return explicitly labelled hashes and correctly recognize the existing remote code; bound failures as above.

4. **High — Publish failure prevents teardown after DONE.** [verda_cli.py:1775](/home/user/bidy2/infra/verda/verda_cli.py:1775)  
   A failed adapter copy raises before deletion. For example, `cp` returning ENOSPC makes every watchdog tick retry publication while the completed GPU keeps billing. Mocked repeated failures confirmed no deletion. **Fix:** Bound publication retries, persist the failure and surviving `runs/<run>/final` location, and delete the GPU; support later publication recovery.

5. **Medium — Switching runs loses the original data identity.** [verda_cli.py:1823](/home/user/bidy2/infra/verda/verda_cli.py:1823), [sft_lora.py:253](/home/user/bidy2/train/sft_lora.py:253)  
   Identity checks cover only the most recent `train_run`. Stop A, run and stop B, change the dataset at the same path, then resume A: the CLI accepts the new hash. The trainer checks argument strings and loads A’s old checkpoint against the changed data. **Fix:** Persist an immutable manifest for every run on the volume and validate it before accepting checkpoints or DONE.

6. **Medium — Training accepts adapters that serving rejects.** [verda_cli.py:2170](/home/user/bidy2/infra/verda/verda_cli.py:2170), [verda_cli.py:986](/home/user/bidy2/infra/verda/verda_cli.py:986)  
   `--lora-r 32` is accepted, while serving always sets maximum rank 16. Training can finish successfully with an unusable serving artifact; [vLLM 0.33 explicitly rejects this mismatch](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/lora/peft_helper.py#L173-L187). **Fix:** Validate effective training arguments, including overrides, and reject ranks above 16 before provisioning.

Validation: The requested pytest command was blocked by the read-only filesystem. All 21 training test functions passed with in-memory persistence and mocked lifecycle locks; additional mocked probes reproduced the failures above. Shell syntax checks passed. No cloud access or file edits.

Verdict: Changes required before unattended overnight training.