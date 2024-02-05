# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

"""This module implements a core class MdRun for molecular dynamics job setup."""
from __future__ import annotations


class MdJob:
    """A core class for MD results analysis."""

    def __init__(self, name):
        """Base constructor."""
        self.name = name

    @classmethod
    def from_dict(cls):
        """
        Constructor.

        Returns:

        """
        return cls("name")

    @classmethod
    def from_recipe(cls):
        """
        Constructor.

        Returns:

        """
        return cls("name")
