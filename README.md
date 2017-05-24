
## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeup_images/test2.jpg "test image2"
[image2]: ./writeup_images/calibration2.jpg "chessboard before"
[image3]: ./writeup_images/undistort2.jpg "Undistorted chessboard"
[image4]: ./writeup_images/undistorted_test2.jpg "Undistorted"
[image5]: ./writeup_images/binary2.bmp "Binary Example"
[image6]: ./writeup_images/straight_lines2_points.jpg "points"
[image7]: ./writeup_images/transformed_image2.jpg "Road Transformed"
[image8]: ./writeup_images/transformed_image2.bmp "Road binary Transformed"
[image9]: ./writeup_images/fitted2.jpg "Fit Visual"
[image10]: ./writeup_images/final_test2.jpg "Output"
[image11]: ./writeup_images/final_test4.jpg "Output2"
[video1]: ./writeup_images/project_video.mp4 "Video"

## Rubric Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README


You're reading it!

### Camera Calibration

#### Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 1st section of the IPython notebook located in "./P4.ipynb" (or in the `_findDistortion()` function in `LaneDetection.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  Lastly, I applied this distortion correction to the test image using the `cv2.undistort()` function.

Here is a test image:


![alt text][image2]


And here is the undistorted image:


![alt text][image3]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
The code for this step is contained in the 2nd section of the IPython notebook located in "./P4.ipynb" (or in the `undistort()` function in `LaneDetection.py`).  

The distortion correction is simply done with the function `cv2.undistort()` using the parameters obtained by `cv2.calibrateCamera()`. 

Here is an example of original road image


![alt text][image1]


And here is an example of the undistorted road image


![alt text][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for this step is contained in the 3rd section of the IPython notebook located in "./P4.ipynb" (or in the `combinedBinary()` function in `LaneDetection.py`).  
In the IPython notebook I tested some different color transform and gradient thresholdings, such as: the R channel of the RGB image; B channel of the LAB transformed image; the L channel of the LUV image; the S channel of the HSV and HLS transformed image; the gradient in x and y directions; the gradient magnitude and angle. I finaly chose to use only the: B channel of the LAB transformed image, the L channel of the LUV image and the gradient in x direction.

Here's an example of my output for this step. 

![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for this step is contained in the 4th section of the IPython notebook located in "./P4.ipynb" (or in the `_setTransformParams()` and `prespectiveTransform()` functions in `LaneDetection.py`).  
  
I draw 4 points on the 2 straight lanes example images, on the lanes themselfs, and made the transform. In an iteration process, I chose to hardcode the source and destination points that cause the lanes in the transformed image to apear straight.

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 457      | 300, 200      | 
| 698, 457      | 300, 700      |
| 1044, 680     | 980, 700      |
| 261, 680      | 980, 200      |

Here are the source points:

![alt text][image6]

Here is the test image after transformation.

![alt text][image7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for this step is partialy contained in the 5th section of the IPython notebook located in "./P4.ipynb" (or fully contained in the `findLane()` and `_fitPoly()` functions in `LaneDetection.py`).  

If there is no history of detected lanes, the lanes are found by this manner:
The base of each lane is found by finding the max valuse i an histogram of the bottom 3rd section of the binary image. Then, the lanes are found in sections using the sliding window search algorithm. If there is history of the lanes, a fast window search is performed, based on a moving average of the previous found lanes.

After the pixels related to the lanes are found, the lanes are fitted with a 2nd degree polinom using the `np.poly` function.

Here is an example of a processed binary image. 


![alt text][image9]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for this step is contained in the 6th section of the IPython notebook located in "./P4.ipynb" (or in the `_findCurve()` and `drawImage()` functions in `LaneDetection.py`).  

First I determined the convertion of pixels to meters for the x & y directions. A new polinom fit in metric units is found, and then the position of the car and the lane curvature are found. The middle of the image is taken as the center of the car, and then I copare this location with the location of the base of the two lanes. The deviation from the middle of the lanes bases and the middle of the image is the position of the car.
 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code for this step is contained in the 7th section of the IPython notebook located in "./P4.ipynb" (or in the `drawImage()` function in `LaneDetection.py`).  

Here is an example of a processed image with identified lanes:

![alt text][image10]

![alt text][image11]

---

### Pipeline (video)

#### Provide a link to your final video output.

Here's a [link to my video result](./output_videos/project_video.mp4)


---

### Discussion

#### Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

In order to make a reliable lane detection pipeline, I created a buffer of previous found lanes. This buffer was updated for each image in the video. If a good detection of lanes was made for the current image, than the lanes parameters are saved in the buffer. I used a buffer with 5 cells. The detection of the lanes was evaluated in order to determine how reliable it is. The conditions that were checked are: 
- the difference in curvature between the 2 lanes
- the road width (distance bewteen the 2 lanes)
- the difference in the fit parametes of the 2 lanes
- 
And for each lans sepratly (if the previous conditions failed):
- the difference in curvature between the found lane and the moving average of the buffer
- the difference in lane base location between the found lane and the moving average of the buffer
- the difference in fit parameters between the found lane and the moving average of the buffer

After the detection of the lane, a moving averages (with decreasing weights) is used to determine the current lane fit parameters.

The main difficulties I faced in this project were related to image processing and reliably thresholding over the images to find the lanes pixels. The challenge videos contain some difficult images with very bad lane marks and images that face the sun or with very sharp turns. For such videos my pipeline will probably fail.

One idea I have to improve the pipeline is to create several pipelines of lane detection, each one corresponding to different image properties (i.e. burned images, bad lanes marks, sharp curves, etc.). Then if one pipeline fails, I can use the others to try and better find the lanes.

