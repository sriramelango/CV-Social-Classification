import gradio as gr
import os
import torch
from PIL import Image

                           
#subprocess.run(["mv","content/custom_data.yaml","./yolov5/data"])         


def load_model():
 '''
 Loading hub model & setting the preferences for the model  
 '''
 model = torch.hub.load('ultralytics/yolov5', 'custom', path='Content/best.pt')
 model.conf = 0.38 
 model.dnn=True
 model.agnostic=True
 return model

model=load_model()
#, force_reload=True
def detect(inp):
 #g = (size / max(inp.size))  #gain
 #im = im.resize((int(x * g) for x in im.size), Image.ANTIALIAS)  # resize 
 results = model(inp,size=640)  # inference
 results.render()  # updates results.imgs with boxes and labels
 return Image.fromarray(results.imgs[0])
 

inp = gr.inputs.Image(type="pil", label="Original Image")
output = gr.outputs.Image(type="pil", label="Output Image")


io=gr.Interface(fn=detect, inputs=inp, outputs=output, title='CV Social Classification',theme='peach')
io.launch(debug=True,share=False)
  
#examples=['Content/4.jpg','Content/10.jpg','Content/18.jpg']

                                             
                                                                                                                                       