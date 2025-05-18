from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import time

model_id = "vikhyatk/moondream2"
revision = "2024-07-23"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

acc = 0

while True:

    try:
        acc = acc + 1
        image = Image.open("detectedPlate" + str(acc) + ".png")
        enc_image = model.encode_image(image)
        print(model.answer_question(enc_image, "read the lisence plate number", tokenizer))
    except FileNotFoundError:
        acc = acc - 1
        print("waiting for files "+str(acc)+"...")
        time.sleep(3)


