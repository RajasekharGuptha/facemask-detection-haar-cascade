# Facemask-Detection-Haar-Cascade
Face mask Detection Haar Cascade file 

Used 102 positive images and 176 negative images

**Reduce size of image before applying cascade for better results** 

For Testing we used 300x300 size for all images 

```python
import cv2 as cv

fmcascade=cv.CascadeClassifier("facemaskcascade.xml")
image = cv.imread("test_image.jpg")
image = cv.resize(image, (300, 300))
fm = fmcascade.detectMultiScale(image, 1.12,2,None,(120,120),(300,300)) # try with different values for better results

for (x, y, w, h) in fm:
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0))

cv.imshow("Final Image",image)

```

If it did not work for any image , convert image to gray mode and then try again  
```python
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

```
Note:- **Not 100% accurate**  
