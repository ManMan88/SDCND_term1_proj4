import numpy as np
import cv2
import glob

class LanesFinder(object):
    def __init__(self,chess_path = './camera_cal/'):
        self.M = []
        self.Minv =[]
        self.distMat =[]
        self.dist =[]
        self._findDistortion(chess_path)
        self._setTransformParams()
    
    def _findDistortion(self,chess_path):
        nx = 9
        ny = 6
        objp = np.zeros((nx*ny,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objPoints = [] # 3d points in real world space
        imgPoints = [] # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob(chess_path + 'calibration*.jpg')

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

        temp = cv2.imread(images[0])
        img_size = temp.shape[0:2]
        ret, mtx, dist, rvecs, tvecs, = cv2.calibrateCamera(objPoints,imgPoints,img_size,None,None)
        self.distMat = mtx
        self.dist = dist
        return 0
    
    def _setTransformParams(self):
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
        return 0
     
    def _binChannel(self,channel,low_thresh=1,high_thresh=255):
        binary_out = np.zeros_like(channel,dtype=np.uint8)
        binary_out[(channel >= low_thresh) & (channel < high_thresh)] = 1
        return binary_out
    
    def _sChannel(self,img,lowThreshold=100,highThreshold=255):
        hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
        sChannel = hls[:,:,2]
        binary_out = self._binChannel(sChannel,lowThreshold,highThreshold)
        return binary_out
    
    def _rChannel(self,img,lowThreshold=230,highThreshold=255):
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
    
    def _findLaneCenter(self,warped,ratio=2/3):
        histogram = np.sum(warped[np.int(warped.shape[0]*ratio):,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        base = np.argmax(histogram)
        return base
    
    def undistort(self,img):
         return cv2.undistort(img,self.distMat,self.dist,None,self.distMat)
       
    def combinedBinary(self,undistorted):
        sBin = self._sChannel(undistorted)
        rBin = self._rChannel(undistorted)
        sobBin = self._xSobel(undistorted) 
        return sBin | rBin | sobBin
        
    def prespectiveTransform(self,img,mat):
        img_size = (img.shape[1],img.shape[0])
        warped = cv2.warpPerspective(img,mat,img_size,flags=cv2.INTER_LINEAR)
        return warped
    
    def findLane(self,warped,lane,mid=0,nwindows=9,margin=100,minpix=50):
        lane_inds = []
        x_ind = []
        y_ind = []
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        if not all(v == 0 for v in lane.baseHistory):
            nonzero = warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            lane_inds = ((nonzerox > (lane.good_fit[0]*(nonzeroy**2) + lane.good_fit[1]*nonzeroy + lane.good_fit[2] - margin)-mid) & (nonzerox < (lane.good_fit[0]*(nonzeroy**2) + lane.good_fit[1]*nonzeroy + lane.good_fit[2] + margin)-mid)) 
            
            # extract left and right line pixel positions
            x_ind = nonzerox[lane_inds]
            y_ind = nonzeroy[lane_inds] 
            lane.pro = 1
        if not any(x_ind) or not any(y_ind):
            lane_inds = []
            base = self._findLaneCenter(warped)           
            nRows = warped.shape[0]
                       
            # Set height of windows
            window_height = np.int(nRows/nwindows)

            # Current positions to be updated for each window
            current = base
            # Create empty lists to receive left and right lane pixel indices
            
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
            lane.pro = 0
        return [x_ind,y_ind]
    
    def checkDetection(self,leftLane,rightLane,curveOffset=500,baseOffset=0.1,fitOffset=1,laneWidth=3.7):
        leftLane.detected = True
        rightLane.detected = True
        if not all(v == 0 for v in leftLane.baseHistory) or not all(v == 0 for v in rightLane.baseHistory):
            if abs((leftLane.current_base + rightLane.current_base)/2 - laneWidth) > 2*fitOffset:
                print('base difference too large')
                if abs(leftLane.movingAverage(leftLane.baseHistory)-leftLane.current_base) >  baseOffset:
                    print('left base offsets too large')
                    leftLane.detected = False
                if abs(rightLane.movingAverage(rightLane.baseHistory)-rightLane.current_base) >  baseOffset:
                    print('right base offsets too large')
                    rightLane.detected = False
            if abs(leftLane.current_curve - rightLane.current_curve) > curveOffset:
                print('curve difference too large',leftLane.current_curve - rightLane.current_curve)
                if abs(leftLane.movingAverage(leftLane.curveHistory)-leftLane.current_curve) > curveOffset:
                    print('left curve offsets too large',abs(leftLane.movingAverage(leftLane.curveHistory)-leftLane.current_curve))
                    leftLane.detected = False
                if abs(rightLane.movingAverage(rightLane.curveHistory)-rightLane.current_curve) > curveOffset:
                    print('right curve offsets too large',abs(rightLane.movingAverage(rightLane.curveHistory)-rightLane.current_curve))
                    rightLane.detected = False
            if (abs(leftLane.current_fit[0] - rightLane.current_fit[0]) > fitOffset) or (abs(leftLane.current_fit[1] - rightLane.current_fit[1]) > fitOffset):
                print('fit[0] difference too large')
                if (abs(leftLane.movingAverage(np.array(leftLane.fitHistory)[:,0])-leftLane.current_fit[0]) > fitOffset) or (abs(leftLane.movingAverage(np.array(leftLane.fitHistory)[:,1])-leftLane.current_fit[1]) > fitOffset):
                    print('left fit offsets too large')
                    leftLane.detected = False
                if (abs(rightLane.movingAverage(np.array(rightLane.fitHistory)[:,0])-rightLane.current_fit[0]) > fitOffset) or (abs(rightLane.movingAverage(np.array(rightLane.fitHistory)[:,1])-rightLane.current_fit[1]) > fitOffset):
                    print('right fit offsets too large')
                    rightLane.detected = False
        else:
            print("no data in history")
        return 0
                
            
        return 0

    def drawImage(self,undistorted,leftLane,rightLane):       
        # Generate x and y values for plotting
        nRows = undistorted.shape[0]
        ploty = np.linspace(0, nRows-1, nRows)
        left_fitx = leftLane.good_fit[0]*ploty**2 + leftLane.good_fit[1]*ploty + leftLane.good_fit[2]
        right_fitx = rightLane.good_fit[0]*ploty**2 + rightLane.good_fit[1]*ploty + rightLane.good_fit[2]
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        color_warp = np.zeros_like(undistorted)
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))
        color_warp[leftLane.current_y,leftLane.current_x,:] = [0,0,255]
        color_warp[rightLane.current_y,rightLane.current_x,:] = [255,0,0]
    
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self.prespectiveTransform(color_warp, self.Minv) 
        # Combine the result with the original image
        final_img = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
        return final_img
    

class Lane(object):
    def __init__(self,history_buf = 5, yEval = 720):
        self.detected = False
        self.current_fit = []
        self.current_base = []
        self.current_curve = []
        self.current_x = []
        self.current_y = []
        self.good_fit = []
        self.baseHistory = []
        self.curveHistory = []
        self.xHistory = []
        self.yHistory = []
        self.fitHistory = []
        self.hisotry_buf = history_buf
        self.y_eval = yEval
        self.weights = [0.5,0.25,0.15,0.7,0.03]
        self.pro = 0

    def _fitPoly(self,x_ind,y_ind,weights=None):
        # Fit a second order polynomial to each
        return np.polyfit(y_ind, x_ind, 2, w=weights)
    
    def _findBase(self,fit,x2m_per_pix=3.7/680):
        return np.polyval(fit,self.y_eval)*x2m_per_pix
    
    def _findCurve(self,x,y,y2m_per_pix=30/720,x2m_per_pix=3.7/680):
        fit_m = self._fitPoly(x2m_per_pix*x,y2m_per_pix*y)
        return ((1 + (2*fit_m[0]*self.y_eval*y2m_per_pix + fit_m[1])**2)**1.5) / np.absolute(2*fit_m[0])
            
    def updateHistory(self):
        if len(self.baseHistory) == self.hisotry_buf:
            self.baseHistory.pop(0)
            self.curveHistory.pop(0)
            self.xHistory.pop(0)
            self.yHistory.pop(0)
            self.fitHistory.pop(0)
        if self.detected:
            self.baseHistory.append(self.current_base)
            self.curveHistory.append(self.current_curve)
            self.xHistory.append(self.current_x)
            self.yHistory.append(self.current_y)
            self.fitHistory.append(self.current_fit)
        else:
            self.baseHistory.append(0)
            self.curveHistory.append(0)
            self.xHistory.append([0])
            self.yHistory.append([0])
            self.fitHistory.append([0,0,0])
        return 0
    
    def addLane(self,x,y):
        self.current_x = x
        self.current_y = y
        self.current_fit = self._fitPoly(x,y)
        self.current_base = self._findBase(self.current_fit)
        self.current_curve = self._findCurve(x,y)
        return 0

    def movingAverage(self,data):
        weight_sum = 0
        data_sum = 0
        for sample,weight in zip(reversed(data),self.weights):
            data_sum += sample*weight
            weight_sum += weight
        return data_sum/weight_sum

    def deriveGoodFit(self):
        if not all(v == 0 for v in self.baseHistory):
            yFit = []
            xFit = []
            weightsFit = []
            for x,y,weight in zip(reversed(self.xHistory),reversed(self.yHistory),self.weights):
                if len(x) > 1:
                    yFit.extend(y)
                    xFit.extend(x)
                    weightsFit.extend(np.ones_like(x)*weight)
            self.good_fit = self._fitPoly(xFit,yFit,weightsFit)
        else:
            self.good_fit = self.current_fit
        return 0
    
    
