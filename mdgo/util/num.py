# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

"""Utilities for manipulating numbers in data structures."""

from __future__ import annotations


def strip_zeros(items: list[str | float | int] | str) -> list[int] | None:
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
