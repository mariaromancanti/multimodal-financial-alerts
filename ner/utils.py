import os
import json


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_vocabularies(token_vocab_path, tag_vocab_path):
    token_to_idx = load_json(token_vocab_path)
    tag_to_idx = load_json(tag_vocab_path)
    idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}
    return token_to_idx, tag_to_idx, idx_to_tag


def files_exist(paths):
    return all(os.path.exists(path) for path in paths)