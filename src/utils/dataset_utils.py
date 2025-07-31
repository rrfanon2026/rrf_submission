import pandas as pd
import logging
import sys
from pathlib import Path
from typing import List, Set

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_into_sets(df: pd.DataFrame, founders_per_set: int = 20, random_shuffle: bool = False) -> list[pd.DataFrame]:
    """
    Splits the input DataFrame into smaller sets of up to `founders_per_set` founders,
    trying to balance successful and unsuccessful founders, but still allowing imbalanced sets
    when not enough are available. A final partial set is added if any founders remain.

    Args:
        df: The input DataFrame containing 'startup_success' column.
        founders_per_set: Maximum number of founders in each set. Defaults to 20.
        random_shuffle: Whether to shuffle the dataset before splitting.

    Returns:
        A list of DataFrames, each representing a set.
    """
    if random_shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    available_successful = df[df['startup_success'] == 1].copy()
    available_unsuccessful = df[df['startup_success'] == 0].copy()

    sets = []
    set_count = 0

    logger.info(f"Attempting to split data into sets of up to {founders_per_set} founders.")

    while len(available_successful) > 0 or len(available_unsuccessful) > 0:
        n_success = min(len(available_successful), founders_per_set // 2)
        n_unsuccess = min(len(available_unsuccessful), founders_per_set - n_success)

        # If not enough total, take more from the other group to reach founders_per_set
        if n_success + n_unsuccess < founders_per_set:
            if len(available_successful) > n_success:
                additional = min(founders_per_set - (n_success + n_unsuccess), len(available_successful) - n_success)
                n_success += additional
            elif len(available_unsuccessful) > n_unsuccess:
                additional = min(founders_per_set - (n_success + n_unsuccess), len(available_unsuccessful) - n_unsuccess)
                n_unsuccess += additional

        if n_success == 0 and n_unsuccess == 0:
            break

        successful_sample = available_successful.sample(n=n_success, random_state=42 + set_count) if n_success > 0 else pd.DataFrame()
        available_successful = available_successful.drop(successful_sample.index)

        unsuccessful_sample = available_unsuccessful.sample(n=n_unsuccess, random_state=42 + set_count) if n_unsuccess > 0 else pd.DataFrame()
        available_unsuccessful = available_unsuccessful.drop(unsuccessful_sample.index)

        combined_set = pd.concat([successful_sample, unsuccessful_sample])
        shuffled_set = combined_set.sample(frac=1, random_state=42 + set_count).reset_index(drop=True)

        sets.append(shuffled_set)
        set_count += 1
        logger.debug(f"Created set {set_count} with {len(shuffled_set)} founders ({n_success} successful, {n_unsuccess} unsuccessful).")

    logger.info(f"Successfully created {len(sets)} sets.")
    return sets

def load_dataset(data_path: str, name: str = "dataset") -> pd.DataFrame:
    """General-purpose loader for any dataset with logging."""
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Successfully loaded {name} with {len(df)} founders.")
        return df
    except Exception as e:
        logger.error(f"Error loading {name}: {e}")
        sys.exit(1)


def load_and_filter(files: List[Path], special: Set[str]) -> pd.DataFrame:
    """
    Load and merge multiple filtered CSV files. Extracts and preserves special rows
    like 'Founder Index' and 'Dataset', and removes any rows matching `special`.

    Args:
        files: List of CSV files to load.
        special: Set of strings to filter out from 'Question' column.

    Returns:
        Combined DataFrame with special rows on top.
    """
    combined, special_rows = [], None
    for fn in files:
        df = pd.read_csv(fn)
        if special_rows is None:
            special_rows = df[df['Question'].isin(['Founder Index','Dataset'])]
        combined.append(df[~df['Question'].isin(special)])
        
    return pd.concat([special_rows]+combined, ignore_index=True)