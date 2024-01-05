from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

dir = os.path.dirname(os.path.abspath(__file__))

for root, dirs, files in os.walk(dir):
    for i, file in enumerate(files):
        if file.endswith('.jpg'):
            img_path = os.path.join(root,file)
            raw_img = Image.open(img_path).convert('RGB')
            text = "This painting depicts " #conditional
            inputs = processor(raw_img, text, return_tensors="pt").to("cuda")
            out = model.generate(**inputs)
            print(processor.decode(out[0], skip_special_tokens=True))

"""
#unconditional
inputs = processor(raw_image, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
"""
