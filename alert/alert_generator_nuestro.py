import json
from pathlib import Path
from typing import List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class AlertGenerator:
    def __init__(
        self,
        model_name: str = "distilgpt2",
        device: str = "cpu",
        backend: str = "hf",
        ollama_url: str = "http://127.0.0.1:11434/api/generate",
    ):
        self.device = device
        self.backend = backend
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.tokenizer = None
        self.model = None

        if self.backend == "hf":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        elif self.backend != "ollama":
            raise ValueError(f"Unsupported alert backend: {backend}")

    @classmethod
    def from_pretrained(cls, model_path: str, device: str = "cpu"):
        instance = cls.__new__(cls)
        instance.device = device
        instance.backend = "hf"
        instance.model_name = model_path
        instance.ollama_url = "http://127.0.0.1:11434/api/generate"
        instance.tokenizer = AutoTokenizer.from_pretrained(model_path)
        instance.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

        if instance.tokenizer.pad_token is None:
            instance.tokenizer.pad_token = instance.tokenizer.eos_token

        instance.model.config.pad_token_id = instance.tokenizer.pad_token_id
        return instance

    def save_pretrained(self, output_dir: str) -> None:
        if self.backend != "hf":
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

    def tokenize_data(
        self,
        prompts: List[str],
        alerts: Optional[List[str]] = None,
        max_length: int = 128,
    ):
        if self.backend != "hf":
            raise RuntimeError("Tokenization is only available for the Hugging Face backend.")

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

    def _build_prompt(self, prompt: str) -> str:
        prompt_with_prefix = prompt.rstrip()
        if not prompt_with_prefix.endswith("Alert:"):
            prompt_with_prefix = f"{prompt_with_prefix}\nAlert:"
        return prompt_with_prefix

    def _generate_with_hf(self, prompt: str, max_new_tokens: int = 40) -> str:
        self.model.eval()
        prompt_with_prefix = self._build_prompt(prompt)

        inputs = self.tokenizer(
            prompt_with_prefix,
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
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        continuation = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if continuation:
            return continuation

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if generated_text.startswith(prompt_with_prefix):
            return generated_text[len(prompt_with_prefix):].strip()

        return generated_text.strip()

    def _generate_with_ollama(self, prompt: str) -> str:
        system_prompt = (
            "You generate a short financial alert in English.\n"
            "Guidelines:\n"
            "1. Base the alert on the entities and sentiment provided in the prompt.\n"
            "2. Do not invent facts that are not supported by the prompt.\n"
            "3. You may mention entity values, labels, or both when helpful.\n"
            "4. If there are several relevant entities, you can include more than three.\n"
            "5. If sentiment is negative, start with 'Financial risk alert:'.\n"
            "6. If sentiment is positive, start with 'Positive financial alert:'.\n"
            "7. If sentiment is neutral, start with 'Informational financial alert:'.\n"
            "8. Keep the output concise and natural, ideally one sentence and at most two.\n"
            "9. Return only the alert text."
        )
        payload = json.dumps(
            {
                "model": self.model_name,
                "prompt": f"{system_prompt}\n\n{self._build_prompt(prompt)}",
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 80,
                    "repeat_penalty": 1.1,
                },
            }
        ).encode("utf-8")

        request = Request(
            self.ollama_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            raise RuntimeError(
                "Could not reach Ollama at http://127.0.0.1:11434. "
                "Make sure Ollama is running and the model is installed."
            ) from exc

        text = data.get("response", "").strip()
        if not text:
            raise RuntimeError("Ollama returned an empty response.")

        return " ".join(text.split())

    def generate_alert(self, prompt: str, max_new_tokens: int = 40) -> str:
        if self.backend == "ollama":
            return self._generate_with_ollama(prompt)
        return self._generate_with_hf(prompt, max_new_tokens=max_new_tokens)
