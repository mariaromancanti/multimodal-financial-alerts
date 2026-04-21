import os
import ollama


def generate_caption(image_path: str, model: str = "llava-phi3") -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No existe la imagen: {image_path}")

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Generate one short natural caption for this image. Only output the caption.",
                "images": [image_path],
            }
        ],
        options={"temperature": 0.2},
    )

    return response["message"]["content"].strip()


if __name__ == "__main__":
    image_path = "imagenes_generadas/train2P.png"
    caption = generate_caption(image_path, model="llava-phi3")
    print("Caption:", caption)
