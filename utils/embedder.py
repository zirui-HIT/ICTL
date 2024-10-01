import sys

from typing import List, Dict, Any
from FlagEmbedding import FlagICLModel

sys.path.append('.')


class Embedder:
    def __init__(self, model_name_or_path: str, config: Dict[str, Any]):
        pass

    def embed(self, source: List[str], config: Dict[str, Any], information: Dict[str, Any]) -> List[List[float]]:
        return [[0] for _ in source]


class FlagICLEmbedder:
    def __init__(self, model_name_or_path: str):
        self.model = FlagICLModel(model_name_or_path,
                                  use_fp16=True)

    def embed(self, source: List[str], config: Dict[str, Any], information: Dict[str, Any] = None) -> List[List[float]]:
        self.model.query_instruction_for_retrieval = information[
            'instruction'] if information and 'instruction' in information else None
        self.model.examples_for_task = information['examples'] if information and 'examples' in information else None
        embeddings = self.model.encode_corpus(source, config['batch_size'])
        return embeddings.tolist()


MODEL_MAP: Dict[str, Embedder] = {
    "bge": FlagICLEmbedder
}


def embed_with_model(model_name_or_path: str, source: List[str], config: Dict[str, Any], information: Dict[str, Any] = None) -> List[List[float]]:
    embedder = MODEL_MAP[model_name_or_path](model_name_or_path, config)
    return embedder.embed(source, config, information)
