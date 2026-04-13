from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class AlertGenerator:
    def __init__(self, model_name: str = "distilgpt2", device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def tokenize_data(
        self,
        prompts: List[str],
        alerts: Optional[List[str]] = None,
        max_length: int = 128,
    ):
        if alerts is None:
            texts = prompts
        else:
            texts = [
                f"{prompt}\nAlert: {alert}"
                for prompt, alert in zip(prompts, alerts)
            ]

        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

        encodings = {key: value.to(self.device) for key, value in encodings.items()}

        if alerts is not None:
            encodings["labels"] = encodings["input_ids"].clone()

        return encodings

    def generate_alert(self, prompt: str, max_new_tokens: int = 40) -> str:
        self.model.eval()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=3,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if generated_text.startswith(prompt):
            return generated_text[len(prompt):].strip()

        return generated_text.strip()
