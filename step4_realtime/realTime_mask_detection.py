#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms as trn
from torch.autograd import Variable as V
import dlib
import cv2


# In[2]:


def preprocess(image):
    center_crop = trn.Compose([
        trn.Resize((256, 256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_input = Image.fromarray(image)
    img_output = V(center_crop(img_input).unsqueeze(0))
    return img_output


# In[3]:


def predict(img_path):
    img=preprocess(img_path)
    outputs = model(img)
    max = torch.softmax(outputs, 1)
    if max[0][0] > 0.5:
      masked = 'not masked'
      label = 0
      prob = max[0][0]
    else:
      masked = 'masked'
      label = 1
      prob = max[0][1]
    return label, prob, masked


# In[4]:


def test_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(img, 1)
    for i, d in enumerate(dets):
        face = img[d.top():d.bottom(),d.left():d.right()]
        rlt, prob, masked = predict(face)

        if 'rlt' in locals():
            if rlt == 1:
                img = cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (0,255,0), 5)
            else:
                img = cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (255,0,0), 5)
            img = cv2.putText(img, (masked + ': {:.3f}'.format(prob)), (d.left(), d.top()), cv2.FONT_HERSHEY_COMPLEX, 1, (150, 150, 150), 2)
    return img


# **Load the model and replace the output layer with a 2 class layer. Then load the save weights from training. Set to evaluate mode and send to cuda if available.**

# In[5]:


model = torchvision.models.mobilenet_v2()
model.classifier[-1] = nn.Linear(in_features=1280, out_features=2, bias=True)
model.load_state_dict(torch.load('mobilenet_v2_masknet.pth', map_location=torch.device('cpu')))
model.eval()
print('')


# **Load the face detection model**

# In[6]:


detector = dlib.get_frontal_face_detector()


# # Test on video
#
#

# In[7]:


cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Mask detection
    frame = test_image(frame)
    cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Stream stopped")


# In[8]:


#  When everything done, release the capture
cap.release()


# In[ ]:
