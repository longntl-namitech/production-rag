import pytest
from test_methods import V010

from production_rag.core.eval.run import parse_args


def test_parse_args_defaults_to_all_variants():
    assert parse_args(["--dataset", "scifact"]).methods == V010


def test_parse_args_resolves_folder_names():
    args = parse_args(["--dataset", "scifact", "--methods", "rerank", "hybrid_rrf"])
    assert args.methods == ["hybrid_rrf_rerank", "hybrid_rrf"]


def test_parse_args_rejects_variantless_folder(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--dataset", "scifact", "--methods", "multi_query"])
    assert "no eval variants" in capsys.readouterr().err
