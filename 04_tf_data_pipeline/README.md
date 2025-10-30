# TensorFlow tf.data Pipeline

Modern, efficient, and production-ready. This is how TensorFlow wants you to load data, and for good reason.

## Why tf.data Exists

TensorFlow created `tf.data` to solve a fundamental problem: your GPU shouldn't wait for data. Traditional approaches load data on the CPU, preprocess it, then send batches to the GPU. During this time, your expensive GPU sits idle.

`tf.data` pipelines overlap data loading, preprocessing, and training. While your GPU trains on batch N, the CPU prepares batch N+1. The result? Up to 3x faster training on the same hardware.

## When To Use This

Use `tf.data` when:
- You're serious about training performance
- Working with medium to large datasets
- Building production ML systems
- Need reproducible data pipelines
- Want to leverage multiple CPU cores efficiently
- Training on GPU/TPU and want to maximize utilization

Stick with simpler approaches if:
- You're just learning and want something straightforward
- Dataset is tiny (< 1000 images) and speed doesn't matter
- You're debugging and prefer simple, synchronous loading
- Not using TensorFlow at all

## Core Concepts

**Datasets are lazy**: `tf.data.Dataset` doesn't load anything until you iterate. It represents a pipeline of operations that will execute when needed.

**Chaining operations**: Build pipelines by chaining transformations. Each step transforms the dataset in some way.

**Automatic optimization**: TensorFlow analyzes your pipeline and applies optimizations automatically. Things like prefetching and parallel processing happen behind the scenes.

**Deterministic by default**: Set a seed and get the same data order every time. Crucial for reproducibility.
