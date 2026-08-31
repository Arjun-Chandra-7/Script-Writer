# Research record and design decisions

Research was conducted before implementation, with preference for official
documentation and primary papers. Accessed 2026-08-31.

## Findings that changed the design

### Drive discovery is reconciled, not trusted as an event stream

Google Drive supports listing files by parent membership and selecting explicit
response fields. Binary files expose `md5Checksum`, but it is not universal.
Drive also supports change notifications, yet notifications are a wake-up
mechanism and require renewal. The architecture consequently uses paginated
full reconciliation as its correctness path and may add notifications only as
an optimization.

- [Search for files and folders](https://developers.google.com/workspace/drive/api/guides/search-files)
- [Drive file resource](https://developers.google.com/workspace/drive/api/reference/rest/v3/files)
- [Notifications for resource changes](https://developers.google.com/workspace/drive/api/guides/push)

### Checkpoint resume is necessary but not sufficient

Hugging Face Trainer can resume from a checkpoint and restores relevant random
states. Its current guidance also describes incomplete-checkpoint sentinels and
graceful termination. The surrounding system must still pin the dataset
snapshot and prevent a failed run from being recreated with different data.

- [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
- [Trainer recipes: resume training](https://huggingface.co/docs/transformers/main/trainer_recipes)

### Exact and near deduplication protect both quality and evaluation

Lee et al. found that duplicated training data increases memorized generation
and contaminates validation data. This supports byte-level SHA-256 admission,
semantic/source grouping before splitting, and a future near-duplicate pass on
transcripts before any training launch.

- [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)

### Naive train-wait-train risks forgetting

Continual fine-tuning can degrade prior capabilities. Research on instruction
continual learning identifies replay as a practical baseline, while results
vary with task similarity and ordering. The design therefore versions adapters,
mixes new examples with a stratified replay buffer, and requires a fixed
regression suite before promotion rather than simply continuing on each batch.

- [An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning](https://arxiv.org/abs/2308.08747)
- [InsCL: A Data-efficient Continual Learning Paradigm for Fine-tuning Large Language Models with Instructions](https://aclanthology.org/2024.naacl-long.37/)
- [Fine-tuned Language Models are Continual Learners](https://aclanthology.org/2022.emnlp-main.410/)

### Parameter-efficient adaptation should be tested, not assumed

LoRA freezes pretrained weights and learns low-rank updates; QLoRA extends this
with a frozen quantized base. These methods make adaptation practical on
limited hardware, but neither guarantees creative quality. A no-training
prompt/retrieval baseline and human evaluation remain mandatory comparators.

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

## Open decisions before training approval

1. Which outcome/analytics files will accompany extractor reports, and what is
   their stable join key?
2. Are all transcripts/scripts owned or licensed for model training?
3. Which niches, languages, platforms, durations, and output schemas are in the
   first model's scope?
4. What GPU/RAM, latency, deployment target, and ongoing budget are available?
5. Who supplies blinded editorial ratings and approves the frozen test set?
6. What minimum batch size/quality threshold triggers a proposed run?
7. Which base-model licenses are acceptable for commercial deployment?

No responsible architecture can promise a perfect or viral model. It can make
data lineage, evaluation, rollback, and continued improvement rigorous enough
that quality claims are measurable instead of assumed.
