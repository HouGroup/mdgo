# coding: utf-8
# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

"""
Utilities for manipulating numbers in data structures.
"""

from typing import List, Union, Optional


def strip_zeros(items: Union[List[Union[str, float, int]], str]) -> Optional[List[int]]:
    """
    Strip the trailing zeros of a sequence.

    Args:
        items: The sequence.

    Return:
        A new list of numbers.
    """
    new_items = [int(i) for i in items]
    while new_items[-1] == 0:
        new_items.pop()
    while new_items[0] == 0:
        new_items.pop(0)
    if len(new_items) == 0:
        return None
    return new_items