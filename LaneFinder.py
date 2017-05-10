import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

class LaneFinder(object):
    def __init__(self,chess_path = './camera_cal/'):
        self.chess_path = chess_path
        self.lane_fit = []
        self.x_vals = []
        self.curve = 0
        self.base = 0
        self.M = []
        self.Minv =[]
        self.distMat =[]
        self.dist =[]
        self._findDistortion(self)
        self._setTransformParams(self)
    
    def _findDistortion(self):
        nx = 9
        ny = 6
        objp = np.zeros((nx*ny,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objPoints = [] # 3d points in real world space
        imgPoints = [] # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob(self.chess_path + 'calibration*.jpg')

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray,(nx,ny),None)
    
        # If found, add object points, image points
        if ret == True:
            objPoints.append(objp)
            imgPoints.append(corners)
        
        ret, mtx, dist, rvecs, tvecs, = cv2.calibrateCamera(objPoints,imgPoints,imgSize,None,None)
        self.distMat = mtx
        self.dist = dist
    
    def __setTransformParams(self):
        # Chosen 4 points
        bot_left = [261,680]
        top_left = [585,457]
        top_right = [698,457]
        bot_right = [1044,680]
        dst_xl = 300
        dst_xr = 980
        dst_yb = 700
        dst_yt = 200
        src = np.float32([bot_left,top_left,top_right,bot_right])
        dst = np.float32([[dst_xl,dst_yb],[dst_xl,dst_yt],[dst_xr,dst_yt],[dst_xr,dst_yb]])
        self.M = cv2.getPerspectiveTransform(src,dst)
        self.Minv = cv2.getPerspectiveTransform(dst,src)
    
    def _undistort(self):
         return cv2.undistort(self.image,self.distMat,self.dist,None,self.distMat)
     
    def _binChannel(self,channel,low_thresh=1,high_thresh=255):
        binary_out = np.zeros_like(channel,dtype=np.uint8)
        binary_out[(channel >= low_thresh) & (channel < high_thresh)] = 1
        return binary_out
    
    def sChannel(self,img,lowThreshold=100,highThreshold=255):
        hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
        sChannel = hls[:,:,2]
        binary_out = self._binChannel(sChannel,lowThreshold,highThreshold)
        return binary_out
    
    def rChannel(self,img,lowThreshold=230,highThreshold=255):
        rChannel = img[:,:,2]
        binary_out = self._binChannel(rChannel,lowThreshold,highThreshold)
        return binary_out
    
    def _xSobel(self,img,lowThreshold=30,highThreshold=255,kernel=11):
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        vChannel = hsv[:,:,2]
        sobel = cv2.Sobel(vChannel,cv2.CV_64F,1,0,ksize=kernel)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint(255*abs_sobel/np.max(abs_sobel))
        binary_out = self._binChannel(scaled_sobel,lowThreshold,highThreshold)
        return binary_out
    
    def _prespectiveTransform(self,img,mat):
        img_size = (img.shape[1],img.shape[0])
        warped = cv2.warpPerspective(img,mat,img_size,flags=cv2.INTER_LINEAR)
        return warped
    
    def _findLaneCenter(self,warped,ratio=2/3):
        histogram = np.sum(warped[np.int(warped.shape[0]*ratio):,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        base = np.argmax(histogram)
        return base
    
    def _findLane(self,warped,nwindows=9,margin=100,minpix=50):
    
        base = self._findLaneCenter(warped)           
        nRows = warped.shape[0]
                   
        # Set height of windows
        window_height = np.int(nRows/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        current = base
        # Create empty lists to receive left and right lane pixel indices
        lane_inds = []
        
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = nRows - (window+1)*window_height
            win_y_high = nRows - window*window_height
            win_x_low = current - margin
            win_x_high = current + margin
 
             
            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
            
            # Append these indices to the lists
            lane_inds.append(good_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_inds) > minpix:
                current = np.int(np.mean(nonzerox[good_inds]))
    
        # Concatenate the arrays of indices
        lane_inds = np.concatenate(lane_inds)
        x_ind = nonzerox[lane_inds]
        y_ind = nonzeroy[lane_inds] 
        return x_ind, y_ind

    def _fitPoly(self,x_ind,y_ind):
        # Fit a second order polynomial to each
        self.lane_fit = np.polyfit(x_ind, y_ind, 2)
        return 0
    

    def _curveNoffset(self,warped,x_ind,y_ind,ym_per_pix=30/720,xm_per_pix = 3.7/700):
        warped = cv2.imread(fname)[:,:,0]
        nRows = warped.shape[0]
        
        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(y_ind*ym_per_pix, x_ind*xm_per_pix, 2)
        
        # Calculate the radius of curvature
        y_eval = nRows-1
        self.curve = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        self.base = np.polyval(self.lane_fit, y_eval)
        
        return 0
        