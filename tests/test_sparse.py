import pytest

from production_rag.core.sparse.bm25 import Bm25Encoder, load_vietnamese_stopwords, strip_stopwords


def test_strip_multi_syllable_stopword():
    assert strip_stopwords("Bởi vì trời mưa", frozenset({"bởi vì"})) == "trời mưa"


def test_strip_longest_phrase_first():
    assert strip_stopwords("bởi vì nắng", frozenset({"bởi", "bởi vì"})) == "nắng"


def test_strip_does_not_cut_inside_syllable():
    assert strip_stopwords("vìa vì", frozenset({"vì"})) == "vìa"


def test_only_stopwords_gives_empty_string():
    assert strip_stopwords("là của", frozenset({"là", "của"})) == ""


def test_stopword_file_loads():
    words = load_vietnamese_stopwords()
    assert len(words) > 1000
    assert any(" " in w for w in words)
    assert not any(w.startswith("#") for w in words)


@pytest.mark.integration
def test_bm25_vi_encoder_handles_all_stopword_query():
    encoder = Bm25Encoder(stopwords=load_vietnamese_stopwords(), disable_stemmer=True)
    [vector] = encoder.encode_queries(["bởi vì là của"])
    assert vector.indices == []
    [doc] = encoder.encode_documents(["thủ tục đăng ký kết hôn"])
    assert doc.indices


def test_strip_handles_nfd_input():
    import unicodedata

    nfc = "Người lao động bởi vì"
    nfd = unicodedata.normalize("NFD", nfc)
    stopwords = frozenset({"bởi vì", "người"})
    assert strip_stopwords(nfd, stopwords) == strip_stopwords(nfc, stopwords) == "lao động"
