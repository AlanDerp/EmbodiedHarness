# EmbodiedHarness operator entrypoint

Use this file when the user asks Codex to run a RoboDojo task category. The standard
Codex instruction filename is `AGENTS.md`. Operate from this repository root.
For code changes, do not start an experiment as a side effect.

## Start a category test

1. Read [.agents/skills/fix-loop/SKILL.md](.agents/skills/fix-loop/SKILL.md) once.
   Use [docs/CODEX_CATEGORY_TEST.zh-CN.md](docs/CODEX_CATEGORY_TEST.zh-CN.md) for initial setup.
2. List the configured categories:
   `embodied-harness test-category --list`.
   The default registry is `configs/categories.local.json`; use `--registry PATH`
   if the operator supplies another deployment. If absent, set up the real local
   configuration from the guide. Never silently run an example or invent a category.
3. Resolve the user's category against the list. Ask only if the choice is ambiguous.
   Run `embodied-harness test-category --category CATEGORY --dry-run` to validate
   the native task, disjoint partition manifest and frozen library without writes,
   service calls, model generation or physics. Report the goal, partition counts
   and output directory before starting.
4. A request to run that category authorizes the configured bounded campaign.
   Run `embodied-harness test-category --category CATEGORY`; it runs `doctor`
   before entering the existing campaign. Do not request confirmation again for
   ordinary configured execution. Missing calibration, model access or services
   are blockers to resolve, never grounds to substitute fixtures.
5. Keep monitoring the foreground process to its terminal outcome. Report
   `campaign.json`, `selected.json`, unseen/heldout successes with full denominators,
   skill promotion and failures. A process ID or successful preflight is not success.
   Preserve interrupted/failed runs; never delete the ledger or automatically
   restart uncertain actions. Use a new explicit run directory for a new experiment.

## Execution boundaries

Codex in this chat is the host operator. Campaign System 2 uses the configured
role service; the bundled `embodied_harness.codex_driver:create` uses fresh Codex
CLI conversations and image attachments. Do not impersonate an independent verifier
using this development chat or share proposer history with it.

Only dev evidence may influence repairs and selection. No edits or retries to
improve unseen/heldout outcomes. No simulator internals, private layout manifests,
scores or old solutions in model contexts. Use public tools, real SAM3 and native
termination plus success. The CLI driver is not an OS isolation boundary: deploy
the model service under a dedicated account/device without private simulator files.

Source changes, CPU tests and fixture runs do not establish real physics success.
Report actual readiness and concrete missing prerequisites honestly.
