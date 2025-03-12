# Messy Handwriting OCR Comparison Between Aya-Vision-8B and Qwen2VL-OCR-2B


![image/png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/3nhG7kJf6RSbm5Dv-fZg0.png)

This comparison is conducted for experimental purposes only. Although this is an unbalanced comparison between an 8B-parameter model and a 2B-parameter OCR model based on Qwen2VL, the goal is to evaluate how lower-parameter models perform in OCR tasks compared to mid-range consumer models like Aya-Vision-8B.  

- You can test both models using the interactive demo available on Hugging Face Spaces

### Models Used in This Comparison  

| Model Name         | Description | Hugging Face Link |
|--------------------|-------------|-------------------|
| *Aya Vision 8B* | An 8B-parameter vision model for OCR and other vision tasks. | [Aya-Vision-8B](https://huggingface.co/CohereForAI/aya-vision-8b) |
| *Qwen2VL-OCR-2B* | A 2B-parameter vision-language model optimized for OCR tasks. | [Qwen2-VL-OCR-2B](https://huggingface.co/prithivMLmods/Qwen2-VL-OCR-2B-Instruct) |
| *Demo Here* | Demo for Aya Vision and Qwen2VL | [Try Here Multimodal OCR](https://huggingface.co/spaces/prithivMLmods/Multimodal-OCR) |

This study explores how **lower-parameter models** handle **messy handwriting recognition** compared to more complex **mid-range vision models**.  

https://github.com/user-attachments/assets/dfe1b581-9185-4e84-b76f-faaf382c0be8

### Experiment Details

The experiment involves analyzing the following images, each categorized by different levels of difficulty in **messy handwriting recognition**. These images serve as test cases for comparing the OCR capabilities of both models.  

### <span style="color:orangered;">Test Sample 1</span>

**Prompt :** OCR the Image

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/eC1pLcRfC61xK8e2P6qPI.jpeg)

| Model              | Extracted Text |
|--------------------|---------------|
| **Aya-Vision-8B** | "The snowman feels positive today. The snowman made himself a large fluffy dog. The show man with his hooves and scarf feels happy and a little girly. You give him a hug. He likes it. He lives in a lovely igloo. He is the very tidy." |
| **Qwen2VL-OCR-2B** | "the snow man feels positive every day, the snow man made him self a large fluffy dog. the snow man is happy with his hat and scarf. he feel happy and gives you a hug. he lives in a lonely igloo. he is very grumpy." |

### <span style="color:orangered;">Test Sample 2</span>

**Prompt :** OCR the Image


![image/png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/_h2sEb1_w8NgC0kTmV-4M.png)

| **Model**            | **Extracted Text** |
|----------------------|------------------|
| **Aya-Vision-8B**   | "Dear Dad,  I am really crediting school. It's making me very hard with all my stuff. I simply can't think of anything I need, so if you want, you could just send me a card. I would love to hear from you.  Love, [Your Name]" |
| **Qwen2VL-OCR-2B**  | "Dear Dad,  School is really great. I am making lots of friends and studying very hard. With all my stuff, I simply can't think of anything I need. So if you want, you can just send me a card, and I would love to hear from you.  Love, Your Son" |

### <span style="color:orangered;">Test Sample 3</span>

**Prompt :** OCR the Image

![image/gif](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/50A8XcG4vkfWFDQ1F9n9Y.gif)

| **Model**            | **Extracted Text** |
|----------------------|------------------|
| **Aya-Vision-8B**   | "NOTE 1  Here goes the text. It's a handwritten note with some scribbles and cursive writing. The text seems to be a mix of thoughts or reminders, possibly related to personal growth or advice. It's a bit difficult to read due to the handwriting style and the quality of the image.  Some key phrases that can be made out include:  - 'Here goes the text'  - 'It's a handwritten note'  - 'Some scribbles and cursive writing'  - 'A mix of thoughts or reminders'  - 'Related to personal growth or advice'  Overall, the note appears to be a personal reflection or a set of thoughts jotted down by hand." |
| **Qwen2VL-OCR-2B**  | "Note 1  Please take these pills three times a day, and you should begin to feel much better." |

### <span style="color:orangered;">Test Sample 4</span>

**Prompt :** OCR the Image

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/vzu876XVHSOV6gnFMB2DR.jpeg)

| **Model**            | **Extracted Text** |
|----------------------|------------------|
| **Aya-Vision-8B**   | "Records. Archaeology has records of sports-like activities in the lives of those cultures and sports played and sports played in the lives of those cultures." |
| **Qwen2VL-OCR-2B**  | "records. Archaeology has a long history of sports-like activities in the cultures and sports played and in the lives of those cultures." |

### <span style="color:orangered;">Test Sample 5</span>

**Prompt :** OCR the Image

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/D5cqcCrX0aAIvD2uAC56v.jpeg)

| **Model**            | **Extracted Text** |
|----------------------|------------------|
| **Aya-Vision-8B**   | "The lion laughed at the mouse and let him go. A few days later, the same lion was caught in a hunter's net. He tried his best to get out of the net but almost succeeded. However, he failed. He roared loudly in rage. The man who heard the roar of the lion and hesitated heard the roar of the lion and hesitated." |
| **Qwen2VL-OCR-2B**  | "The lion laughed at the mouse and let him go. A few days later, the same lion was caught in a hunter's net. He tried his almost to get out of the net but failed. He roared loudly in rage. The mouse heard the roar of the lion and hastened." |

### <span style="color:orangered;">Test Sample 6</span>

**Prompt :** OCR the Image

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/F3GLLqDpFe0co-jO6bolM.jpeg)

| **Model**            | **Extracted Text** |
|----------------------|------------------|
| **Aya-Vision-8B**   | "Today is Thursday, October 20th. But it definitely feels like a Friday. I'm making a second cup of coffee, and I've even finished my notes. Sometimes I flip through older notes, and my handwriting is unrecognizable. I've tried writing in all caps, but it looks forced and unnatural. I'll just take notes on my laptop or paper. I'm not sure what pen I use. I've tried different types of pens, but none seem to be perfect. I feel stressed at what I've written. People write this!" |
| **Qwen2VL-OCR-2B**  | "Today is Thursday, October 30th - But it definitely feels like a Friday. I'm already considering making a second cup of coffee - and I haven't even finished my first. Do I have a problem? Sometimes I'll flip through older notes I've taken, and my handwriting is unrecognizable. Perhaps it depends on the type of pen I use? I've tried writing in all caps but it looks so forced and unnatural. Often times, I'll just take notes on my laptop, but I still seem to gravitate toward pen and paper. Any advice on what to imprave? I already feel stressed out looking at what I've just written - it looks like 3 different people wrote this!!" |

### <span style="color:orangered;">Test Sample 7</span>

**Prompt :** OCR the Image Content

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/GRXeHsiLekEVRK0Z5uOSW.jpeg)

| **Model**            | **Extracted Text** |
|----------------------|------------------|
| **Aya-Vision-8B**   | "The A.M. Palmer Co., New York City.  Gentlemen,  I have completed the lessons in the Palmer Method of Business Writing, and herewith submit my examination. I have tried to follow closely the printed instructions in the manual, and hope to obtain a final certificate.  Awaiting your decision,  Sam,  Sincerely." |
| **Qwen2VL-OCR-2B**  | "Your Town, Jan 30, 1915.  The A. N. Palmer Co., New York City.  Gentlemen: - I have completed the lessons in the Palmer Method of Business Writing, and herewith submit my examination. I have tried to follow closely the printed instructions in the manual, and hope to obtain a Final Certificate.  Awaiting your decision, I am,  Sincerely." |

### <span style="color:orangered;">Test Sample 8</span>

**Prompt :** OCR the Image

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/4HqfK-KDYAnXmqCtlG_E5.jpeg)

| **Model**            | **Extracted Text** |
|----------------------|------------------|
| **Aya-Vision-8B**   | "Tired with all there, for restful death cry, As to behold a bigger born, and needy nothing thrimmed in joy, and gilded faith unhappily forlorn, and maiden honours shamefully strained, and sight and perfection as sungfully disgusted and strength by limping sway and art made strong and art like doctor-like controlling child and captive good attending captain it would I be gone, I leave my love alone." |
| **Qwen2VL-OCR-2B**  | "Tired with all these, for restful death I cry, As to behold desert a bigger born, And needy nothing tremid'd in jollity, And purest faith unhappily forsorn, And gilded honours shamefully misplaced, And maiden virtue rudely trumpeted, And sight perfection wrongfully disgraced, And strength by limping sway disabled, And art mode tongue-tied by authority, And folly doctor-like controlling skill, And captive good attending captain ill. Tired with all these from these would I be gone, Fare that, to die, I leave my lose alone." |

### <span style="color:orangered;">Test Sample 9</span>

**Prompt :** OCR the Image

![image/png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/FC_80IrTjnfa4IEidjIsP.png)

| **Model**            | **Extracted Text** |
|----------------------|------------------|
| **Aya-Vision-8B**   | "The British gave Independence to India on 15th August 1947. Indians celebrate the Independence Day with great joy. 26th January is India's Republic Day. Independence Day and Republic Day are celebrated as national festivals and we have holidays on these days." |
| **Qwen2VL-OCR-2B**  | "The British gave Independence to India on 15th August 1947. Indians celebrate the Independence day with great joy. January 26th is India's Republic day. Independence day and Republic day are celebrated as national festivals and we have holidays on these days." |

### <span style="color:orangered;">Test Sample 10</span>

**Prompt :** OCR the Image

![image/png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/wPdfkljtOxQVKNQUR3FOV.png)

| **Model**            | **Extracted Text** |
|----------------------|------------------|
| **Aya-Vision-8B**   | "This is a great tool for people like me who have terrible handwriting." |
| **Qwen2VL-OCR-2B**  | "This is a great tool for people like me who have terrible handwriting." |


### Quickstart with transformers🤗

```py
!pip install -q gradio_client==1.3.0  
!pip install -q qwen-vl-utils==0.0.2  
!pip install -q transformers-stream-generator==0.0.4  
!pip install -q torch==2.4.0  
!pip install -q torchvision==0.19.0  
!pip install -q git+https://github.com/huggingface/transformers.git@v4.49.0-AyaVision  
!pip install -q accelerate  
!pip install -q av  
!pip install -q gradio  
```

```py
import gradio as gr
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TextIteratorStreamer,
    AutoModelForImageTextToText,
)
from transformers.image_utils import load_image
from threading import Thread
import time
import torch
import spaces
from PIL import Image
import requests
from io import BytesIO

# Helper function to return a progress bar HTML snippet.
def progress_bar_html(label: str) -> str:
    return f'''
<div style="display: flex; align-items: center;">
    <span style="margin-right: 10px; font-size: 14px;">{label}</span>
    <div style="width: 110px; height: 5px; background-color: #FFB6C1; border-radius: 2px; overflow: hidden;">
        <div style="width: 100%; height: 100%; background-color: #FF69B4 ; animation: loading 1.5s linear infinite;"></div>
    </div>
</div>
<style>
@keyframes loading {{
    0% {{ transform: translateX(-100%); }}
    100% {{ transform: translateX(100%); }}
}}
</style>
    '''

QV_MODEL_ID = "prithivMLmods/Qwen2-VL-OCR-2B-Instruct" # or use  #prithivMLmods/Qwen2-VL-OCR2-2B-Instruct
qwen_processor = AutoProcessor.from_pretrained(QV_MODEL_ID, trust_remote_code=True)
qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
    QV_MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to("cuda").eval()

AYA_MODEL_ID = "CohereForAI/aya-vision-8b"
aya_processor = AutoProcessor.from_pretrained(AYA_MODEL_ID)
aya_model = AutoModelForImageTextToText.from_pretrained(
    AYA_MODEL_ID, device_map="auto", torch_dtype=torch.float16
)

@spaces.GPU
def model_inference(input_dict, history):
    text = input_dict["text"].strip()
    files = input_dict.get("files", [])
    
    if text.lower().startswith("@aya-vision"):
        # Remove the command prefix and trim the prompt.
        text_prompt = text[len("@aya-vision"):].strip()
        if not files:
            yield "Error: Please provide an image for the @aya-vision feature."
            return
        else:
            # For simplicity, use the first provided image.
            image = load_image(files[0])
            yield progress_bar_html("Processing with Aya-Vision-8b")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt},
                ],
            }]
            inputs = aya_processor.apply_chat_template(
                messages,
                padding=True,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(aya_model.device)
            # Set up a streamer for Aya-Vision output
            streamer = TextIteratorStreamer(aya_processor, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(
                inputs, 
                streamer=streamer, 
                max_new_tokens=1024, 
                do_sample=True, 
                temperature=0.3
            )
            thread = Thread(target=aya_model.generate, kwargs=generation_kwargs)
            thread.start()
            buffer = ""
            for new_text in streamer:
                buffer += new_text
                buffer = buffer.replace("<|im_end|>", "")
                time.sleep(0.01)
                yield buffer
            return

    # Load images if provided.
    if len(files) > 1:
        images = [load_image(image) for image in files]
    elif len(files) == 1:
        images = [load_image(files[0])]
    else:
        images = []
    
    # Validate input: require both text and (optionally) image(s).
    if text == "" and not images:
        yield "Error: Please input a query and optionally image(s)."
        return
    if text == "" and images:
        yield "Error: Please input a text query along with the image(s)."
        return

    # Prepare messages for the Qwen2-VL model.
    messages = [{
        "role": "user",
        "content": [
            *[{"type": "image", "image": image} for image in images],
            {"type": "text", "text": text},
        ],
    }]
    
    prompt = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = qwen_processor(
        text=[prompt],
        images=images if images else None,
        return_tensors="pt",
        padding=True,
    ).to("cuda")
    
    # Set up a streamer for real-time output.
    streamer = TextIteratorStreamer(qwen_processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1024)
    
    # Start generation in a separate thread.
    thread = Thread(target=qwen_model.generate, kwargs=generation_kwargs)
    thread.start()
    
    buffer = ""
    yield progress_bar_html("Processing with Qwen2VL OCR")
    for new_text in streamer:
        buffer += new_text
        buffer = buffer.replace("<|im_end|>", "")
        time.sleep(0.01)
        yield buffer

examples = [
    [{"text": "@aya-vision Summarize the letter", "files": ["examples/1.png"]}],
    [{"text": "@aya-vision Extract JSON from the image", "files": ["example_images/document.jpg"]}],
    [{"text": "Extract as JSON table from the table", "files": ["examples/4.jpg"]}],
    [{"text": "@aya-vision Describe the photo", "files": ["examples/3.png"]}],
    [{"text": "@aya-vision Summarize the full image in detail", "files": ["examples/2.jpg"]}],
    [{"text": "@aya-vision Describe this image.", "files": ["example_images/campeones.jpg"]}],
    [{"text": "@aya-vision What is this UI about?", "files": ["example_images/s2w_example.png"]}],
    [{"text": "Can you describe this image?", "files": ["example_images/newyork.jpg"]}],
    [{"text": "Can you describe this image?", "files": ["example_images/dogs.jpg"]}],
    [{"text": "@aya-vision Where do the severe droughts happen according to this diagram?", "files": ["example_images/examples_weather_events.png"]}],
]

demo = gr.ChatInterface(
    fn=model_inference,
    description="# **Multimodal OCR `@aya-vision 'prompt..'`**",
    examples=examples,
    textbox=gr.MultimodalTextbox(
        label="Query Input", 
        file_types=["image"], 
        file_count="multiple", 
        placeholder="By default, it runs Qwen2VL OCR, Tag @aya-vision for Aya Vision 8B"
    ),
    stop_btn="Stop Generation",
    multimodal=True,
    cache_examples=False,
)

demo.launch(debug=True)
```

This is an experimental test to improve **prithivMLmods/Qwen2-VL-OCR-2B-Instruct** and **prithivMLmods/Qwen2-VL-OCR2-2B-Instruct** for better OCR performance. The **Qwen2-VL-OCR-2B-Instruct** model performs neck and neck with **Aya-Vision 8B** in some cases and even outperforms it in certain OCR tasks.  

Share your thoughts on the sample cases!  

Thanks for reading!  🤗 







