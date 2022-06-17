# Copyright 2018-2022 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
"""Create an Image Classification Web App using PyTorch and Streamlit."""
# import libraries
from email.mime import image
from PIL import Image
import torch
from torchvision import models, transforms
import streamlit as st
import torchxrayvision as xrv

from tensorflow.keras.utils import img_to_array





# set title of app
st.title("MultiClass Classification Application")
st.write("")


col1, col2 = st.columns(2)
with col1:
    st.header("Please provide an image to anaylze")


# enable users to upload images for the model to make predictions
file_up = st.file_uploader("Upload an image", type = ["jpg", "png","jpeg"])




with col2:
    st.header("Predictions")
    st.image("https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png", width=200)





def predict(image):
    """Return top 5 predictions ranked by highest probability.

    Parameters
    ----------
    :param image: uploaded image
    :type image: jpg
    :rtype: list
    :return: top 5 predictions ranked by highest probability
    """
    # create a ResNet model
    model = xrv.baseline_models.chexpert.DenseNet()

    # transform the input image through resizing, normalization
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
            ),
        transforms.Grayscale(num_output_channels=1)
        ])



    # load the image, pre-process it, and make predictions

    img = Image.open(image)
    batch_t = torch.unsqueeze(transform(img), 0)
    model.eval()
    out = model(batch_t)


    # return the top 5 predictions ranked by highest probabilities
    prob = torch.nn.functional.softmax(out, dim = 1)[0]*1000
    _, indices = torch.sort(out, descending = True)
    return [(xrv.datasets.default_pathologies[idx], prob[idx].item()) for idx in indices[0][:]]


if file_up is not None:
    # display image that user uploaded
    image = Image.open(file_up)
    st.image(image, caption = 'Uploaded Image.', use_column_width = True)
    st.write("")
    st.write("Just a second ...")
    labels = predict(file_up)

    # print out the top 5 prediction labels with scores
    for i in labels:
        st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])






