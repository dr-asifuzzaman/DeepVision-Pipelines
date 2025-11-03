# PyTorch DataLoader Pipeline

PyTorch's approach to data loading. If you're building with PyTorch instead of TensorFlow, this is your home.

## PyTorch vs TensorFlow Philosophy

PyTorch and TensorFlow handle data differently:

**PyTorch**: Pythonic and explicit. You write Python classes that define how data loads. Debugging feels natural because it's just Python code.

**TensorFlow**: Graph-based and optimized. You build pipelines that TensorFlow optimizes automatically. More magic, potentially faster.

Neither is better - they're different tools for different preferences. This pipeline shows the PyTorch way.

## When To Use PyTorch DataLoader

Use PyTorch DataLoader when:
- Your model is built in PyTorch
- You prefer explicit, Pythonic code
- Working in research where flexibility matters
- Need easy debugging with standard Python tools
- Team is familiar with PyTorch ecosystem

Stick with TensorFlow if:
- Using TensorFlow/Keras for modeling
- Need maximum performance optimization
- Working with TPUs (PyTorch support is limited)
- Deploying to TensorFlow Serving

## Core Components

PyTorch data loading has two main parts:

**Dataset**: Defines how to access individual samples
- Implements `__len__()` and `__getitem__()`
- Just returns one sample at a time
- No batching, no shuffling

**DataLoader**: Handles batching, shuffling, parallel loading
- Wraps your Dataset
- Creates batches
- Shuffles data
- Loads in parallel with multiple workers