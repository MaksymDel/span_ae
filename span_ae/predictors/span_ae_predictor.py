
from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('span_ae')
class SpanAePredictor(Predictor):
    """"Predictor wrapper for the Span Based Autoencoder"""

    def load_line(self, line: str) -> JsonDict:  # pylint: disable=no-self-use
        """
        If your inputs are not in JSON-lines format (e.g. you have a CSV)
        you can override this function to parse them correctly.
        """
        json_dict = {"src": line}
        return sanitize(json_dict)

    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        top_span_ids = outputs["top_spans"]

        reconstructed_sentence = " ".join(outputs["predicted_tokens"])

        return reconstructed_sentence

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        sentence = json_dict['src']
        instance = self._dataset_reader.text_to_instance(source_string=sentence)

        return instance, {}
