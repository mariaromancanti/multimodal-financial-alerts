import argparse
import json
import os
import re

from image_captioning import generate_caption

SECTION_IMAGE_INFO = {
    "TRAIN_POSITIVE": ("train", "P"),
    "TRAIN_NEGATIVE": ("train", "N"),
}


def parse_selected_examples(txt_path: str):
    entries = []
    section = None
    item_pattern = re.compile(r"^\s*(\d+)\.\s*(.*\S)\s*$")

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped in SECTION_IMAGE_INFO:
                section = stripped
                continue

            if section is None:
                continue

            match = item_pattern.match(line)
            if match:
                index = int(match.group(1))
                text = match.group(2).strip()
                entries.append((section, index, text))

    return entries


def build_image_path(images_dir: str, section: str, index: int):
    prefix, suffix = SECTION_IMAGE_INFO[section]
    base_name = f"{prefix}{index}{suffix}"
    for ext in (".png", ".jpeg", ".jpg"):
        candidate = os.path.join(images_dir, f"{base_name}{ext}")
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"No existe ninguna imagen esperada en '{images_dir}' con base '{base_name}' "
        "y extensión .png/.jpeg/.jpg"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Genera un JSON con texto y caption para cada imagen correspondiente."
    )
    parser.add_argument(
        "--txt",
        default="imagenes_generadas/selected_sa_examples.txt",
        help="Ruta al archivo selected_sa_examples.txt",
    )
    parser.add_argument(
        "--images-dir",
        default="imagenes_generadas",
        help="Directorio donde están las imágenes",
    )
    parser.add_argument(
        "--output",
        default="imagenes_generadas/captions.json",
        help="Ruta de salida del JSON",
    )
    parser.add_argument(
        "--model",
        default="llava-phi3",
        help="Modelo configurado para generar captions",
    )
    args = parser.parse_args()

    entries = parse_selected_examples(args.txt)
    result = []

    for section, index, text in entries:
        image_path = build_image_path(args.images_dir, section, index)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"No existe la imagen esperada: {image_path}")

        caption = generate_caption(image_path, model=args.model)

        result.append({"text": text, "caption": caption})

    with open(args.output, "w", encoding="utf-8") as out_file:
        json.dump(result, out_file, ensure_ascii=False, indent=2)

    print(f"JSON generado en: {args.output}")


if __name__ == "__main__":
    main()
