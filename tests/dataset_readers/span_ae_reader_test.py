# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common.util import ensure_list
from allennlp.data.fields import ListField, SpanField

from span_ae import SpanAeDatasetReader


class TestSpanAeDatasetReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_default_format(self, lazy):
        reader = SpanAeDatasetReader(lazy=lazy)
        instances = reader.read('tests/fixtures/parallel_copy.tsv')
        instances = ensure_list(instances)

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields["source_tokens"].tokens] == ["@@START@@", "this", "is",
                                                                    "a", "sentence", "@@END@@"]
        assert [t.text for t in fields["target_tokens"].tokens] == ["@@START@@", "this", "is",
                                                                    "a", "sentence", "@@END@@"]
        fields = instances[1].fields
        assert [t.text for t in fields["source_tokens"].tokens] == ["@@START@@", "this", "is",
                                                                    "another", "@@END@@"]
        assert [t.text for t in fields["target_tokens"].tokens] == ["@@START@@", "this", "is",
                                                                    "another", "@@END@@"]
        fields = instances[2].fields
        assert [t.text for t in fields["source_tokens"].tokens] == ["@@START@@", "all", "these", "sentences",
                                                                    "should", "get", "copied", "@@END@@"]
        assert [t.text for t in fields["target_tokens"].tokens] == ["@@START@@", "all", "these", "sentences",
                                                                    "should", "get", "copied", "@@END@@"]

    def test_source_add_start_token(self):
        reader = SpanAeDatasetReader(source_add_start_token=False)
        instances = reader.read('tests/fixtures/parallel_copy.tsv')
        instances = ensure_list(instances)

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields["source_tokens"].tokens] == ["this", "is", "a", "sentence", "@@END@@"]
        assert [t.text for t in fields["target_tokens"].tokens] == ["@@START@@", "this", "is",
                                                                    "a", "sentence", "@@END@@"]

    def test_spans_work_correctly(self):
        reader = SpanAeDatasetReader(max_span_width=1)
        instances = reader.read('tests/fixtures/parallel_copy.tsv')
        instances = ensure_list(instances)
        assert len(instances) == 3

        fields = instances[0].fields
        assert type(fields["source_spans"]) == ListField
        assert type(fields["source_spans"].field_list[0]) == SpanField

        assert len(fields["source_spans"].field_list) == len(fields["source_tokens"].tokens)

        reader = SpanAeDatasetReader(max_span_width=2)
        instances = reader.read('tests/fixtures/parallel_copy.tsv')
        instances = ensure_list(instances)
        fields = instances[1].fields
        assert len(fields["source_spans"].field_list) == len(fields["source_tokens"].tokens) * 2 - 1

        reader = SpanAeDatasetReader(max_span_width=3)
        instances = reader.read('tests/fixtures/parallel_copy.tsv')
        instances = ensure_list(instances)
        fields = instances[1].fields
        assert len(fields["source_spans"].field_list) == len(fields["source_tokens"].tokens) * 3 - 3

        # assert [t.text for t in fields["source_tokens"].tokens] == ["@@START@@", "this", "is", "a", "sentence", "@@END@@"]
        # assert [t.text for t in fields["target_tokens"].tokens] == ["@@START@@", "this", "is",
        #                                                             "a", "sentence", "@@END@@"]

