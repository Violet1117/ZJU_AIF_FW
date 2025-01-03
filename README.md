# ZJU_AIF_FW
This is a final work for Fundamentals of Artificial Intelligence.

## Dataset Structure

### Introduction
**Fer2013** contains approximately 30,000 facial RGB images of different expressions with size restricted to 48×48, and the main labels of it can be divided into 7 types: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral. The Disgust expression has the minimal number of images – 600, while other labels have nearly 5,000 samples each.

|0    |1      |2   |3    |4  |5       |6      |
|-----|-------|----|-----|---|--------|-------|
|Angry|Disgust|Fear|Happy|Sad|Surprise|Neutral|

### Download
The dataset of this project is downloaded by [kaggle](https://www.kaggle.com/). ([FER2013](https://www.kaggle.com/datasets/deadskull7/fer2013/data))

```shell
#! if never use kaggle
pip install kaggel
#!/bin/bash
kaggle datasets download deadskull7/fer2013
unzip fer2013.zip -d dataset
```

### The csv file

|emotion|pixels |Usage   |
|-------|-------|----|
|0~6    |48*48<br>(string stored)|Training(80%)<br> PrivateTest(10%)<br> PublicTest(10%)|

## Streamlit
Using streamlit, we can qiuckly biuld web apps.

To run this project:
```shell
#! if you do not install streamlit
pip install streamlit

#! To run the app
streamlit run home.py
```

Because this is a web app, so there are some limitaions I have not addressed:
- When you train the model, you **cannot** do anything else on this web app. Otherwise, the current page you are visiting would change when you navigate to another page or adjust the model's parameters.
- The running logic of the web app is that whenever you make a change, the page (.py) would run again from the beginning. So there may be some bugs when you do something unexpected.
