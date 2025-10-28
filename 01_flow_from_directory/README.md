# Flow From Directory Pipeline

The simplest way to load images for deep learning. If your images are organized in folders by class, this is your starting point.

## When To Use This

You should use this approach when:
- Your images are already organized in class-based folders
- You're prototyping quickly and need something that works immediately
- You don't need to track additional metadata beyond class labels
- You have a standard multi-class classification problem

Skip this if:
- Your images aren't organized by folder structure
- You need to track metadata (age, quality scores, timestamps, etc.)
- You're working with multi-label problems
- You need regression targets instead of classification

## How It Works

Keras's `ImageDataGenerator` scans your directory structure and automatically:
- Detects class names from folder names
- Assigns numeric labels to each class
- Creates batches of images with their labels
- Applies augmentation on-the-fly during training
