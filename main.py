import os

# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import tensorflow_hub as hub

from neural_style_transfer import StyleStealer

import gradio as gr

if __name__ == "__main__":
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    worker = StyleStealer(hub_model)

    demo = gr.Interface(
        fn=worker.steal,
        inputs=["image", "image"],
        outputs=["image"]
    )

    demo.launch(share=True)
