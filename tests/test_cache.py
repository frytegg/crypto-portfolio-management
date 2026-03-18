"""Smoke tests for the diskcache layer."""

from __future__ import annotations

import tempfile
import time

import diskcache
import pytest


@pytest.fixture
def tmp_cache(tmp_path):
    """Create a temporary diskcache instance for testing."""
    return diskcache.Cache(str(tmp_path / "test_cache"), size_limit=10_000_000, timeout=1)


class TestCacheBasics:
    """Basic cache operations."""

    def test_set_and_get(self, tmp_cache: diskcache.Cache) -> None:
        tmp_cache.set("key1", "value1")
        assert tmp_cache.get("key1") == "value1"

    def test_get_missing_returns_none(self, tmp_cache: diskcache.Cache) -> None:
        assert tmp_cache.get("nonexistent") is None

    def test_ttl_expiry(self, tmp_cache: diskcache.Cache) -> None:
        tmp_cache.set("ephemeral", "data", expire=0.1)
        assert tmp_cache.get("ephemeral") == "data"
        time.sleep(0.2)
        assert tmp_cache.get("ephemeral") is None

    def test_delete(self, tmp_cache: diskcache.Cache) -> None:
        tmp_cache.set("to_delete", "value")
        tmp_cache.delete("to_delete")
        assert tmp_cache.get("to_delete") is None

    def test_stores_dataframe(self, tmp_cache: diskcache.Cache, sample_returns) -> None:
        """Verify diskcache can serialize/deserialize pandas DataFrames."""
        tmp_cache.set("returns", sample_returns)
        recovered = tmp_cache.get("returns")
        assert recovered is not None
        assert recovered.shape == sample_returns.shape
        assert list(recovered.columns) == list(sample_returns.columns)


class TestCacheGetOrFetch:
    """Test the cache_get_or_fetch pattern."""

    def test_cache_hit_skips_fetch(self, tmp_cache: diskcache.Cache) -> None:
        tmp_cache.set("cached_key", 42, expire=3600)
        call_count = 0

        def expensive_fn():
            nonlocal call_count
            call_count += 1
            return 99

        # Simulate cache_get_or_fetch logic
        value = tmp_cache.get("cached_key")
        if value is None:
            value = expensive_fn()
            tmp_cache.set("cached_key", value, expire=3600)

        assert value == 42
        assert call_count == 0

    def test_cache_miss_calls_fetch(self, tmp_cache: diskcache.Cache) -> None:
        call_count = 0

        def expensive_fn():
            nonlocal call_count
            call_count += 1
            return 99

        value = tmp_cache.get("missing_key")
        if value is None:
            value = expensive_fn()
            tmp_cache.set("missing_key", value, expire=3600)

        assert value == 99
        assert call_count == 1
