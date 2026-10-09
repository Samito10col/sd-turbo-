import torch
from diffusers import AutoPipelineForText2Image

# Detecta automáticamente el hardware disponible
if torch.cuda.is_available():
    dispositivo = "cuda"   # tarjeta NVIDIA
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    dispositivo = "xpu"    # gráfica Intel
else:
    dispositivo = "cpu"    # procesador (más lento)

print(f"Usando: {dispositivo}")
print("Cargando modelo...")

modelo = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float32
    # float32 = más precisión y estable; float16 = más rápido pero puede dar imagen negra
)
modelo = modelo.to(dispositivo)

prompt = input("Escribe el prompt de la imagen: ")
negative_prompt = "low quality, worst quality, blurry, jpeg artifacts, bad anatomy, bad hands, extra fingers, mutated hands, disfigured face, deformed eyes, long neck, watermark, text, logo, signature"

print("Generando la imagen...")

imagen = modelo(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=10,  # número de veces que corrige la imagen
    guidance_scale=2.0,      # qué tan fiel será al prompt
    height=512,
    width=512,
).images[0]

imagen.save("imagen.png")
print("Imagen guardada!")
