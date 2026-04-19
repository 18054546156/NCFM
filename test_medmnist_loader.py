#!/usr/bin/env python3
"""
Test script to verify MedMNIST data loading works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.medmnist_dataset import load_medmnist_data
import torch

def test_dataset(dataset_name):
    print(f"\n{'='*60}")
    print(f"Testing {dataset_name}")
    print(f"{'='*60}")

    try:
        train_data, test_data = load_medmnist_data(dataset_name, download=True)

        print(f"\n✓ Train set loaded successfully!")
        print(f"  Shape: {train_data.images.shape}")
        print(f"  Classes: {train_data.nclass}")

        print(f"\n✓ Test set loaded successfully!")
        print(f"  Shape: {test_data.images.shape}")

        # Verify data range
        print(f"\n✓ Data range: [{train_data.images.min():.3f}, {train_data.images.max():.3f}]")

        # Count unique labels
        unique_train = torch.unique(train_data.targets)
        unique_test = torch.unique(test_data.targets)
        print(f"\n✓ Unique labels in train: {unique_train.tolist()}")
        print(f"✓ Unique labels in test: {unique_test.tolist()}")

        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("MedMNIST Loader Test")
    print("="*60)

    # Test balanced dataset
    success1 = test_dataset('pathmnist')

    # Test imbalanced dataset
    success2 = test_dataset('dermamnist')

    if success1 and success2:
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ Some tests failed!")
        print("="*60)
