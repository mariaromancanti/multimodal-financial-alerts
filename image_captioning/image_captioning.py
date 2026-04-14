import os
import ollama


def generate_caption_with_ollama(
    image_path: str,
    model: str = "llama3.2-vision",
    max_words: int = 18,
) -> str:
    """
    Genera un caption para una imagen usando un modelo multimodal de Ollama.

    Args:
        image_path: ruta de la imagen.
        model: modelo multimodal de Ollama.
        max_words: longitud máxima deseada del caption.

    Returns:
        Caption generado como string.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No existe la imagen: {image_path}")

    prompt = f"""
You are an image captioning system.
Generate ONE concise, natural caption for the image.

Rules:
- Output only the caption.
- No lists.
- No explanations.
- No quotes.
- Maximum {max_words} words.
- Focus on the main visible scene.
"""

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_path],
            }
        ],
        options={
            "temperature": 0.2,
        }
    )

    caption = response["message"]["content"].strip()
    return caption


if __name__ == "__main__":
    image_path = "imagenes_generadas/train2P.png"
    caption = generate_caption_with_ollama(image_path)
    print("Caption:", caption)