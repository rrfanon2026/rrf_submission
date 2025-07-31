"""
This file contains utility functions that are used in the project.
"""

from typing import TypeVar, List, Optional, Dict, Any, Union
from datetime import datetime
from dateutil import parser

from rapidfuzz import fuzz
import requests

from core import settings


T = TypeVar("T")


def google_search(q: str, num_results: int) -> List[Dict[str, Any]]:
    """
    Search google for the given query.

    Args:
        q (str): The query to search for.
        num_results (int): The number of search results to return. Max 15.
    
    Raises:
        AssertionError: If the number of results is greater than 15.

    Returns:
        List[Dict[str, Any]]: A list of search results.
    """
    assert num_results <= 15, "Number of results must be less than or equal to 15"

    params = {"query": q, "num_results": num_results}
    headers = {"accept": "application/json", "serp-vela-key": settings.SERP_VELA_KEY}

    response = requests.get(settings.SERP_VELA_URL + "search/", params=params, headers=headers)
    response.raise_for_status()
    data = response.json().get("results", [])
    return data


def number_to_money(number: Union[float, int]) -> str:
    """
    Converts a number to a money string.
    If in the thousands, it will be formatted with K.
    If in the millions, it will be formatted with M.
    If in the billions, it will be formatted with B.
    If in the trillions, it will be formatted with T.
    If in the quadrillions, it will be formatted with Q.
    Else, it will be returned as is as a string.

    Args:
        number (Union[float, int]): The number to convert.

    Returns:
        str: The money string.
    """
    if number is None:
        return ""
    if number < 10**3:
        return str(number)
    if number < 10**6:
        return f"{number/10**3:.1f}K"
    if number < 10**9:
        return f"{number/10**6:.1f}M"
    if number < 10**12:
        return f"{number/10**9:.1f}B"
    if number < 10**15:
        return f"{number/10**12:.1f}T"
    if number < 10**18:
        return f"{number/10**15:.1f}Q"
    return str(number)


def match_strings(str1: str, str2: str, threshold: float = 0.85, non_alphanumeric: bool = True, case_insensitive: bool = True) -> bool:
    """
    Compares two strings and returns True if their similarity ratio is above the given threshold.

    Args:
        str1 (str): The first string to compare.
        str2 (str): The second string to compare.
        threshold (float, optional): The minimum similarity ratio required for the strings to be considered a match. Defaults to 0.85.
        non_alphanumeric (bool, optional): Whether to remove non-alphanumeric characters before comparing. Defaults to True.
        case_insensitive (bool, optional): Whether to ignore case when comparing. Defaults to True.

    Returns:
        bool: True if the similarity ratio is above the threshold, False otherwise.
    """
    assert all(
        [isinstance(str1, str), isinstance(str2, str)]
    ), "Both inputs must be strings."
    str1_, str2_ = str1, str2

    if case_insensitive:
        str1_, str2_ = str1_.lower(), str2_.lower()

    if non_alphanumeric:
        str1_ = "".join([c for c in str1_ if c.isalnum()])
        str2_ = "".join([c for c in str2_ if c.isalnum()])

    return fuzz.ratio(str1_, str2_) > (threshold * 100)


def str_to_std_datetime(datetime_: Union[str, datetime]) -> Optional[datetime]:
    """
    Converts a string to a datetime object.

    Parameters:
        datetime_ (Union[str, datetime]): The datetime to be converted.

    Returns:
        datetime | None: The converted datetime or None if the conversion fails.
    """
    try:
        if isinstance(datetime_, int) or isinstance(datetime_, float):
            datetime_ = str(datetime_)
        if isinstance(datetime_, datetime):
            return datetime_

        return parser.parse(datetime_)
    except (ValueError, TypeError):
        return None


def camel_split(s: str) -> str:
    """
    Split a camel case string into words.
    Example:
        >>> camel_split("camelCaseString")
        "camel Case String"
    
    Args:
        s (str): The camel case string to split.
        
    Returns:
        str: The split string.
    """
    return "".join([" " + i if i.isupper() else i for i in s]).strip()

