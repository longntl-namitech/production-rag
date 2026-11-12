import pytest
from pydantic import ValidationError

from production_rag.core.models import SearchRequest


def test_request_defaults_to_one_default_spec():
    req = SearchRequest(query="q", collection="c")
    assert [s.name for s in req.specs] == ["default"]
    assert req.specs[0].search_type == "hybrid"
    assert req.subqueries == ["q"]


def test_duplicate_spec_names_rejected():
    with pytest.raises(ValidationError):
        SearchRequest(query="q", collection="c", specs=[{"name": "a"}, {"name": "a"}])


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        SearchRequest(query="q", collection="c", unknown=1)


def test_collection_name_pattern():
    with pytest.raises(ValidationError):
        SearchRequest(query="q", collection="a/b")


def test_query_list_limits():
    assert SearchRequest(query=["a", "b"], collection="c").subqueries == ["a", "b"]
    with pytest.raises(ValidationError):
        SearchRequest(query=[f"q{i}" for i in range(9)], collection="c")
    with pytest.raises(ValidationError):
        SearchRequest(query=[], collection="c")
    with pytest.raises(ValidationError):
        SearchRequest(query="", collection="c")
