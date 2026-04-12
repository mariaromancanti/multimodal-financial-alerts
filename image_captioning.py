from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

# 1. Cargar modelo preentrenado
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 2. Cargar imagen
# image_path = "imagen_prueba.jpg"  # cambia esto
image_path = "prueba_texto.jpeg"  
image = Image.open(image_path).convert("RGB")

# 3. Preprocesado
pixel_values = processor(images=image, return_tensors="pt").pixel_values
pixel_values = pixel_values.to(device)

# 4. Generar caption
output_ids = model.generate(pixel_values, max_length=50, num_beams=4)
caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Caption:", caption)