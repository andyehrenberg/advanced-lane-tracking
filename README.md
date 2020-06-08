**Advanced Lane Finding Project**

![GIF](sample.gif)

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: other_outputs/undistort.png "Undistorted"
[image2]: other_outputs/undistortedroad.png "Road Transformed"
[image3]: other_outputs/detect_edges.png "Binary Example"
[image4]: other_outputs/detect_edges2.png "Another Example"
[image5]: other_outputs/warp.png "Warp Example"
[image6]: other_outputs/windows.png "Windows"
[image7]: other_outputs/roi.png "Region of Interest"
[image8]: other_outputs/overlay.png "Output"
[image9]: other_outputs/metrics.png "With Metrics"
[video1]: project_video_output.mp4 "Video"

### Part 1: Camera Calibration

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines 18 through 35 of `helpers.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. I go through each chessboard image in a directory to get the object pounds and image points.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Part 2: Finding Lane Lines

#### Step 1. Undistort the Image

Given the parameters calculated during the camera calibration, I undistort the image using the function in lines 37 - 38 of `helpers.py`.

![alt text][image2]

#### Step 2. Apply a combination of color thresholding and gradients to create a binary image that shows edges.

My function to create a binary image is on lines 67 through 114 in `helpers.py`. I look for parts of the image with a high value in the l and s channels in an HLS version of the image, and high gradient values in a grayscale version. Here's an example of my output for this step.

![alt text][image3]
![alt text][image4]

#### Step 3. Perspective transform

In cell 7 of `advanced_lane_tracking.ipynb`, I define the desired mapping for my perspective transform, which is meant to give a bird's eye view of the lane lines. The matrices M and M_inv are later used to transform the image to bird's eye view, and then back to the camera's original perspective.

```python
bottom_left = [200,720]
bottom_right = [1110, 720]
top_left = [560, 470]
top_right = [722, 470]
src = np.float32([bottom_left,bottom_right,top_right,top_left])

bottom_left = [320,720]
bottom_right = [920, 720]
top_left = [310, 1]
top_right = [920, 1]
dst = np.float32([bottom_left,bottom_right,top_right,top_left])

M = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| [200,720]      | [320,720]       | 
| [1110, 720]     | [920, 720]      |
| [560, 470]    | [310, 0]      |
| [722, 470]     | [920, 0]        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### Step 4: Identifying and Interpolating Lane Lines 

In the transformed binary image, I detected parts of the image where pixels are non-zero, and created a histogram to represent which horizontal regions contained large amounts of non-zero pixels. Spikes in the histogram helped guide where lanes lines were. 
I stored locations of non-zero pixels within windows that started at the bottom of the (transformed) image, and followed the mean x values of non-zero pixels up to the top of the image. I then fit second order polynomials to these "good" pixels to interpolate the lane lines. In subsequent steps, I didn't use the sliding windows approach, and instead searched for non-zero pixels within a region of interest centered around the previous lines. When this approach would fail to find lines, I would revert to the sliding windows method. Lines 148 through 345 of `helpers.py` contain the relevant functions.

![alt text][image6]
![alt text][image7]

#### Step 5: Calculating Radius of Curvature and Distance from Center of the Lane

Lines 391 through 412 of `helpers.py` show my implentations to find these values. It was important to keep in mind the conversions between pixel space and actual space in order to have the resulting outputs be in terms of meters.

#### Step 6: Transform Back to Original Perspective and Fill in the Lane

See lines 418 through 429 of `helpers.py` to see my implementation. Essentially, I take the fitted x values of the left and right lane lanes, combine them with a range of y values, stack these sets of values, and fill the space between them. Then, I convert back to the original perspective of the camera. Here's an example output:

![alt text][image8]
![alt text][image9]

---

### Putting it All Together

I combined all of these steps, and made some tweaks to create (somewhat) smooth lane predictions for videos. At first, we are searching from scratch, so we use the sliding windows approach to find the lanes. We can then usually use the area of interest search, but if this approach fails, we then return to searching from scratch. Former lines are stored, and the current line is a weighted average of the most recent polynomial fit and recent lines. The code is in the function "pipeline" in block 13 of `advanced_lane_tracking.ipynb`.

Here's a [link to my video result](project_video_output.mp4)

---

### Discussion

#### 1. This solution suffered a bit on the challenge video due to lack of robustness towards shadows, and change in the color of the tarmac. Using previous lines more effectively may help this, but I think that this mostly hand designed approach can’t generalize very well. I believe that a learning-based approach will be more effective down the line.  
