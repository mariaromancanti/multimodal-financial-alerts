import json

def load_data(file_path):
    tokens_list = []
    tags_list = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            tokens = example["tokens"]
            ner_tags = example["ner_tags"]

            if len(tokens) != len(ner_tags):
                continue

            tokens_list.append(tokens)
            tags_list.append(ner_tags)

    return tokens_list, tags_list