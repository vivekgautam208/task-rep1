# MoE Training Runtime — Correctness & Recovery

A machine-learning systems project focused on diagnosing silent training errors in mixture-of-experts (MoE) models. The challenge is to keep model updates reproducible when training uses shared parameters, multiple microbatches, mixed precision, sharpness-aware optimization, and checkpoint recovery.

## The Problem

A training run can report a reasonable loss while applying the wrong parameter update. This becomes especially difficult to debug when several model paths share the same weights or a job is interrupted between stages of an optimizer update.

This project explores the engineering work needed to make training mathematically correct and recoverable, without relying on a large GPU cluster.

## Key Areas

- **Gradient Correctness:** Residual model graphs, shared or overlapping weights, and objectives spanning microbatches.
- **Optimizer Behavior:** Gradient accumulation, mixed-precision overflow, clipping, AdamW, EMA, and SAM/ASAM updates.
- **Fault Recovery:** Checkpoints, worker reassignment, repeated requests, and interruptions during updates.
- **Reproducibility:** Deterministic synthetic workloads and numerical comparisons against an independent reference.

## Project Structure

The task package contains the training environment, task instructions, verification tests, runtime configuration, and reference implementation.

## Running the Sample Workload

From the task's `environment` directory, run:

`python3 train_moe.py data/visible_workload.json output.json`

The starter is intentionally incomplete. Producing a report alone does not demonstrate correctness; model state, gradients, and recovery behavior must also agree with the expected results.

## Data and Verification

The workloads are **synthetic and deterministic**, designed to represent realistic distributed-training failures on small CPU-friendly examples.

Verification checks both output reports and training state, including recovery after interruptions. Small floating-point differences are tolerated, but a different training objective or optimizer update is not.

## Project Context

**Category:** Machine Learning

**Subcategory:** Training

This work is relevant to ML infrastructure engineers maintaining large-scale training pipelines, particularly where silent state divergence can waste compute or undermine reproducibility.

**Sharing Note:** If this task belongs to a private evaluation or benchmark program, protected tests, reference solutions, and other restricted files should remain private. Only authorized materials should be published.
