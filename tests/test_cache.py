from production_rag.core.cache import DiskCache


def test_cache_roundtrip_and_miss(tmp_path):
    cache = DiskCache(tmp_path)
    key = DiskCache.key("m", "prompt", "text")
    assert cache.get(key) is None
    cache.put(key, "a short context")
    assert cache.get(key) == "a short context"
    assert DiskCache(tmp_path).get(key) == "a short context"


def test_cache_key_changes_with_model_prompt_and_text():
    base = DiskCache.key("m", "p", "t")
    assert len({base, DiskCache.key("m2", "p", "t"), DiskCache.key("m", "p2", "t")}) == 3
    assert DiskCache.key("m", "p", "t2") != base
    # the parts cannot be shifted into each other
    assert DiskCache.key("ab", "c", "d") != DiskCache.key("a", "bc", "d")


def test_cache_corrupt_file_is_a_miss(tmp_path):
    cache = DiskCache(tmp_path)
    key = DiskCache.key("m", "p", "t")
    cache.put(key, "ok")
    (tmp_path / key[:2] / f"{key}.json").write_text("{not json")
    assert cache.get(key) is None


def test_cache_keeps_unicode(tmp_path):
    cache = DiskCache(tmp_path)
    key = DiskCache.key("m", "p", "Điều 1")
    cache.put(key, "Đoạn này nói về nghỉ phép.")
    assert cache.get(key) == "Đoạn này nói về nghỉ phép."
