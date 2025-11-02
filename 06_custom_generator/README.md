# Custom Generator Pipeline

When nothing else fits your needs, build it yourself. Complete control over data loading, preprocessing, and batching.

## What Are Custom Generators?

A custom generator is a Python function or class that yields batches of data on demand. Instead of using pre-built tools like ImageDataGenerator or tf.data, you write the loading logic yourself.

This means you control:
- Exactly how files are read from disk
- Preprocessing steps and their order
- Augmentation techniques
- Batch composition
- Memory management
- Everything else

## When To Use Custom Generators

Build a custom generator when:
- Your data format is unusual (medical imaging, satellite data, etc.)
- You need preprocessing steps not available in standard tools
- Working with multiple data sources simultaneously
- Implementing research papers with specific data requirements
- Legacy code requires a specific interface
- Need to integrate external libraries (OpenCV, PIL, custom C++ code)

Don't build one if:
- Standard tools work fine for your use case
- You're just starting out and want simplicity
- Team needs maintainable, documented code
- Performance is critical (tf.data is usually faster)

The power of custom generators is flexibility. The cost is maintenance and potential performance issues if not implemented carefully.
