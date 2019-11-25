import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import bdcn
import runway
from runway.data_types import *

@runway.setup(options={"checkpoint" : file(extension=".pth")})
def setup(opts):
    model = bdcn.BDCN()
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(opts["checkpoint"]))
        model.cuda()
    else:
        model.load_state_dict(torch.load(opts["checkpoint"], map_location='cpu'))
    model.eval()

    return model

command_inputs = {"input_image" : image}
command_outputs = {"output_image" : image}

@runway.command("detect_edges", inputs=command_inputs, outputs=command_outputs, description="Detect edges in given image")
def detect_edges(model, inputs):
    img = np.array(inputs["input_image"], np.float32)

    mean_bgr = np.array([104.00699, 116.66877, 122.67892])
    img -= mean_bgr
    img = img.transpose((2, 0, 1))
    data = torch.from_numpy(img).float().unsqueeze(0)
    if torch.cuda.is_available():
        data = data.cuda()
    data = Variable(data)
    out = model(data)

    out = [F.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]]

    out_img = 255 - (255*out[-1])

    return {"output_image" : out_img}

if __name__ == "__main__":
    runway.run(model_options={"checkpoint" : "final-model/bdcn_pretrained_on_bsds500.pth"})
