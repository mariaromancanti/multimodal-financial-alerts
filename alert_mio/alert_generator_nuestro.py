# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch


# class AlertGenerator:
#     def __init__(self, model_name="distilgpt2", device="cpu"):
#         self.device = device

#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

#         # GPT2 no tiene pad_token por defecto
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token

#         self.model.config.pad_token_id = self.tokenizer.pad_token_id

#     def tokenize_data(self, prompts, alerts=None, max_length=128):
#         """
#         Tokeniza prompts y, si existen, también las alertas objetivo.
#         Sirve tanto para entrenamiento como para inferencia.
#         """
#         encodings = self.tokenizer(
#             prompts,
#             truncation=True,
#             padding=True,
#             max_length=max_length,
#             return_tensors="pt"
#         )

#         encodings = {key: value.to(self.device) for key, value in encodings.items()}

#         if alerts is not None:
#             labels = self.tokenizer(
#                 alerts,
#                 truncation=True,
#                 padding=True,
#                 max_length=max_length,
#                 return_tensors="pt"
#             )["input_ids"].to(self.device)

#             encodings["labels"] = labels

#         return encodings

#     def generate_alert(self, prompt, max_new_tokens=40):
#         """
#         Genera una alerta a partir de un prompt.
#         """
#         self.model.eval()

#         inputs = self.tokenizer(
#             prompt,
#             return_tensors="pt",
#             truncation=True,
#             padding=True
#         ).to(self.device)

#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=max_new_tokens,
#                 num_beams=5,
#                 early_stopping=True,
#                 pad_token_id=self.tokenizer.pad_token_id
#             )

#         generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return generated_text
    




from transformers import AutoModelForCausalLM, AutoTokenizer


class AlertGenerator:
    def __init__(self, model_name="distilgpt2", device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_alert(self, prompt, max_length=80):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id
        )

        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        return generated_text