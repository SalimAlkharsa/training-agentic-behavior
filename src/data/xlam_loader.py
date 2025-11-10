"""
xLAM-60k Dataset Loader

Loads the Salesforce xLAM function-calling dataset from HuggingFace.
Provides utilities for downloading, caching, and splitting the dataset.
"""

import os
from typing import Dict, Tuple, Optional
from datasets import load_dataset, Dataset, DatasetDict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XLAMDatasetLoader:
    """
    Loader for the Salesforce xLAM function-calling-60k dataset.

    The dataset contains:
    - 60,000 function-calling examples
    - 3,673 real-world APIs across 21 categories
    - Fields: query (str), tools (list), answers (list)
    """

    DATASET_NAME = "Salesforce/xlam-function-calling-60k"
    DEFAULT_CACHE_DIR = "./data/xlam"

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the dataset loader.

        Args:
            cache_dir: Directory to cache the downloaded dataset.
                      Defaults to ./data/xlam
        """
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.dataset = None

    def load(self, split: Optional[str] = None) -> Dataset:
        """
        Load the xLAM dataset from HuggingFace.

        Args:
            split: Which split to load ('train', 'test', etc.).
                  If None, loads all available splits.

        Returns:
            Dataset or DatasetDict with the loaded data
        """
        logger.info(f"Loading xLAM dataset from {self.DATASET_NAME}...")

        try:
            if split:
                self.dataset = load_dataset(
                    self.DATASET_NAME,
                    split=split,
                    cache_dir=self.cache_dir
                )
                logger.info(f"Loaded {len(self.dataset)} examples from '{split}' split")
            else:
                self.dataset = load_dataset(
                    self.DATASET_NAME,
                    cache_dir=self.cache_dir
                )
                logger.info(f"Loaded dataset with splits: {list(self.dataset.keys())}")

            return self.dataset

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def create_splits(
        self,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        seed: int = 42
    ) -> DatasetDict:
        """
        Create train/validation/test splits from the dataset.

        Args:
            train_size: Fraction of data for training (default 0.8)
            val_size: Fraction of data for validation (default 0.1)
            test_size: Fraction of data for test (default 0.1)
            seed: Random seed for reproducibility

        Returns:
            DatasetDict with 'train', 'validation', and 'test' splits
        """
        if self.dataset is None:
            logger.info("Dataset not loaded, loading now...")
            self.load()

        # Verify split sizes sum to 1.0
        total = train_size + val_size + test_size
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Split sizes must sum to 1.0, got {total} "
                f"(train={train_size}, val={val_size}, test={test_size})"
            )

        # Get the raw dataset (handle both Dataset and DatasetDict)
        if isinstance(self.dataset, DatasetDict):
            # If already split, use the 'train' split or first available
            dataset = self.dataset.get('train') or self.dataset[list(self.dataset.keys())[0]]
        else:
            dataset = self.dataset

        logger.info(f"Creating splits: train={train_size}, val={val_size}, test={test_size}")

        # First split: train vs. (val + test)
        train_test_split = dataset.train_test_split(
            test_size=(val_size + test_size),
            seed=seed
        )

        # Second split: val vs. test
        val_test_ratio = val_size / (val_size + test_size)
        val_test_split = train_test_split['test'].train_test_split(
            test_size=(1 - val_test_ratio),
            seed=seed
        )

        # Create final DatasetDict
        splits = DatasetDict({
            'train': train_test_split['train'],
            'validation': val_test_split['train'],
            'test': val_test_split['test']
        })

        logger.info(f"Split sizes: train={len(splits['train'])}, "
                   f"val={len(splits['validation'])}, test={len(splits['test'])}")

        return splits

    def save_splits(self, splits: DatasetDict, output_dir: str):
        """
        Save the dataset splits to disk.

        Args:
            splits: DatasetDict containing train/validation/test splits
            output_dir: Directory to save the splits
        """
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Saving splits to {output_dir}")
        splits.save_to_disk(output_dir)
        logger.info("Splits saved successfully")

    def load_splits(self, input_dir: str) -> DatasetDict:
        """
        Load previously saved dataset splits from disk.

        Args:
            input_dir: Directory containing the saved splits

        Returns:
            DatasetDict with loaded splits
        """
        logger.info(f"Loading splits from {input_dir}")
        splits = DatasetDict.load_from_disk(input_dir)
        logger.info(f"Loaded splits: {list(splits.keys())}")
        return splits

    def get_dataset_info(self) -> Dict:
        """
        Get information about the loaded dataset.

        Returns:
            Dictionary with dataset statistics and metadata
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        # Handle both Dataset and DatasetDict
        if isinstance(self.dataset, DatasetDict):
            dataset = self.dataset[list(self.dataset.keys())[0]]
        else:
            dataset = self.dataset

        # Get first example for structure inspection
        first_example = dataset[0]

        info = {
            'total_examples': len(dataset),
            'fields': list(first_example.keys()),
            'example_query': first_example.get('query', 'N/A'),
            'example_tools_count': len(first_example.get('tools', [])),
            'example_answers_count': len(first_example.get('answers', []))
        }

        return info


def main():
    """
    Example usage of the XLAMDatasetLoader.
    """
    # Initialize loader
    loader = XLAMDatasetLoader()

    # Load dataset
    dataset = loader.load()

    # Print dataset info
    info = loader.get_dataset_info()
    print("\n=== Dataset Info ===")
    for key, value in info.items():
        print(f"{key}: {value}")

    # Create splits
    splits = loader.create_splits(train_size=0.8, val_size=0.1, test_size=0.1, seed=42)

    # Save splits
    loader.save_splits(splits, output_dir="./data/splits")

    # Show first example from training set
    print("\n=== First Training Example ===")
    first_train = splits['train'][0]
    print(f"Query: {first_train['query']}")
    print(f"Number of tools: {len(first_train['tools'])}")
    print(f"Number of answers: {len(first_train['answers'])}")


if __name__ == "__main__":
    main()
