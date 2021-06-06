import streamlit as st
from fastai.vision.all import *

def load_learner_(path):
    return load_learner(path)

def load_img(path):
    image = Image.open(path)
    w, h = image.size
    dim = (500, int((h*500)/w))
    return image.resize(dim)

learn = load_learner_('export.pkl')

st.markdown("# Fruit Classifier")
st.markdown("Upload an image and the classifier will tell you whether its rotten, ripe or unripe fruit.")
im=[]
for i in range(0,5):
  file_bytes = st.file_uploader("Upload a file", type=("png", "jpg", "jpeg", "jfif"))
  if file_bytes:
    img =load_img(file_bytes)
    st.image(img)
    im=im+[img]
    
    submit = st.button('Predict!')
    if submit:
      probabilities=[]
      predictions=[]
      for j in im:
        pred, pred_idx, probs = learn.predict(im[j])
        probability=float(probs[pred_idx])
        probabilities=probabilities+[probability]
        predictions=predictions+[pred]
      RottenCount=0
      UnripeCount=0
      RottenProbab=[]
      UnripeProbab=[]
      RipeProbab=[]
      for i in range(0,5):
        if("Rotten" in predictions[i]):
          RottenCount+=1
          RottenProbab=RottenProbab+[probabilities[i]]
          RottenPred=predictions[i]
        elif("Unripe" in predictions[i]):
          UnripeCount+=1
          UnripeProbab=UnripeProbab+[probabilities[i]]
          UnripePred=predictions[i]
        else:
          RipeProbab=RipeProbab+[probabilities[i]]
          RipePred=predictions[i]
      if RottenCount>=1:
        avg=sum(RottenProbab)/len(RottenProbab)
        pred=RottenPred
  
      elif UnripeCount>=2:
        avg=sum(UnripeProbab)/len(UnripeProbab)
        pred=UnripePred

      else:
        avg=sum(RipeProbab)/len(RipeProbab)
        pred=RipePred
      st.markdown(f'Prediction: **{pred}**; Probability: **{avg:.04f}**')
