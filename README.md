# PROJECT
## Aim
To write a python program using OpenCV to do the following image manipulations.
i) Extract ROI from  an image.
ii) Perform handwritting detection in an image.
iii) Perform object detection with label in an image.
## Software Required:
Anaconda - Python 3.7
## Algorithm:
## I)Perform ROI from an image
### Step1:
Import necessary packages 
### Step2:
Read the image and convert the image into RGB
### Step3:
Display the image
### Step4:
Set the pixels to display the ROI 
### Step5:
Perform bit wise conjunction of the two arrays  using bitwise_and 
### Step6:
Display the segmented ROI from an image.
## II)Perform handwritting detection in an image
### Step1:
Import necessary packages 
### Step2:
Define a function to read the image,Convert the image to grayscale,Apply Gaussian blur to reduce noise and improve edge detection,Use Canny edge detector to find edges in the image,Find contours in the edged image,Filter contours based on area to keep only potential text regions,Draw bounding boxes around potential text regions.
### Step3:
Display the results.
## III)Perform object detection with label in an image
### Step1:
Import necessary packages 
### Step2:
Set and add the config_file,weights to ur folder.
### Step3:
Use a pretrained Dnn model (MobileNet-SSD v3)
### Step4:
Create a classLabel and print the same
### Step5:
Display the image using imshow()
### Step6:
Set the model and Threshold to 0.5
### Step7:
Flatten the index,confidence.
### Step8:
Display the result.
## Program:
## I)Perform ROI from an image
```python
#In[41]:Import necessary packages 
import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[42]:Read the image and convert the image into RGB
image_path = 'flower.jpg'
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# In[44]:Display the image
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')
plt.show()


# In[45]:Set the pixels to display the ROI 
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_yellow = np.array([22, 93, 0])#choose the RGB values accordingly to display specific color
upper_yellow = np.array([45, 255, 255])
mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)


# In[48]:Perform bit wise conjunction of the two arrays  using bitwise_and 
segmented_image = cv2.bitwise_and(img, img, mask=mask)


# In[49]:Convert the image from BGR2RGB
segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)


# In[50]:Display the segmented ROI from an image.
plt.imshow(segmented_image_rgb)
plt.title('Segmented Image (Yellow)')
plt.axis('off')
plt.show()
```
## II)Perform handwritting detection in an image
```python
# In[1]:


get_ipython().system('pip install opencv-python numpy matplotlib')


# In[2]:Import necessary packages 


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[3]:Define a function to read the image,Convert the image to grayscale,
#Apply Gaussian blur to reduce noise and improve edge detection,
#Use Canny edge detector to find edges in the image,
#Find contours in the edged image,
#Filter contours based on area to keep only potential text regions,
#Draw bounding boxes around potential text regions.
def detect_handwriting(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detector to find edges in the image
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to keep only potential text regions
    min_area = 100
    text_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Draw bounding boxes around potential text regions
    img_copy = img.copy()
    for contour in text_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title('Handwriting Detection')
    plt.axis('off')
    plt.show()

# Path to the image containing handwriting
image_path = 'images.png'

# Perform handwriting detection
detect_handwriting(image_path)
```
## III)Perform object detection with label in an image
```python
# In[2]:Import necessary packages 
import cv2
import matplotlib.pyplot as plt

# In[4]:Set and add the config_file,weights to ur folder.
config_file='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model='frozen_inference_graph.pb'

# In[5]:Use a pretrained Dnn model (MobileNet-SSD v3)
model=cv2.dnn_DetectionModel(frozen_model,config_file)

# In[6]:Create a classLabel and print the same
classLabels = []
file_name='Labels.txt'
with open(file_name,'rt')as fpt:
    classLabels=fpt.read().rstrip('\n').split('\n')

# In[7]:Print the classLabels
print(classLabels)

# In[8]:
print(len(classLabels))

# In[9]:Display the image using imshow()
img=cv2.imread('download.jpeg')
plt.imshow(img)


# In[10]:
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[11]:Set the model and Threshold to 0.5
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)#255/2=127.5
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)
ClassIndex,confidence,bbox=model.detect(img,confThreshold=0.5)
print(ClassIndex)


# In[29]:Flatten the index,confidence.
font_scale=3
font=cv2.FONT_HERSHEY_PLAIN
for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
    cv2.rectangle(img,boxes,(0,0,255),2)
    cv2.putText(img,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(255,0,0),thickness=1)


# In[30]:Display the result.
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
```
## I)Perform ROI from an image:
![og img](https://github.com/vishal21004/project/assets/119560110/e06a8b22-fc1e-4b94-b86d-6449683c7ff7)
![seg img](https://github.com/vishal21004/project/assets/119560110/ae79bce9-20a7-4d77-9013-1b8f3030adeb)



## II)Perform handwritting detection in an image:
![hnd img](https://github.com/vishal21004/project/assets/119560110/10e734a3-a188-4c24-8295-5d7839ad31f1)


## III)Perform object detection with label in an image:
![obj img](https://github.com/vishal21004/project/assets/119560110/ed1b484c-5922-4bbe-a03d-a5bea05ded11)
![lst img](https://github.com/vishal21004/project/assets/119560110/faead252-79da-428f-b51c-148eb15b8d1e)

## Result :
Thus a python program using OpenCV is written to do the following image manipulations:

i) Extract ROI from an image. ii) Perform handwritting detection in an image. iii) Perform object detection with label in an image.
