"""Shared test settings. Hypothesis runs derandomized by default (the same examples on every run, so a result can be
repeated) and writes no example database into the repository; CA_HYPOTHESIS_PROFILE=deep draws 20,000 fresh random
examples per property instead."""

import os

import pytest
from hypothesis import settings

settings.register_profile("ca", derandomize=True, database=None, deadline=None, max_examples=300)
settings.register_profile("deep", database=None, deadline=None, max_examples=20_000)
settings.load_profile(os.environ.get("CA_HYPOTHESIS_PROFILE", "ca"))


def pytest_collection_modifyitems(config, items):
    """Tests marked `shapiq` run only with CA_TEST_SHAPIQ=1: they take 5–7 minutes on a login node."""
    if os.environ.get("CA_TEST_SHAPIQ") == "1":
        return
    skip = pytest.mark.skip(reason="needs shapiq (5–7 minutes in all); CA_TEST_SHAPIQ=1 runs it")
    for item in items:
        if "shapiq" in item.keywords:
            item.add_marker(skip)
