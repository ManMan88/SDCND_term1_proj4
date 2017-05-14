import LaneDetection as ld
from moviepy.editor import VideoFileClip

def process_image(img):
    # pipeline
    #img = cv2.imread('./straight_lines1.jpg') # ADD! take image from video
    #plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    undistorted = lf.undistort(img)
    #plt.imshow(cv2.cvtColor(undistorted,cv2.COLOR_BGR2RGB))
    binary = lf.combinedBinary(undistorted)
    #plt.imshow(binary,cmap='gray')
    warped = lf.prespectiveTransform(binary,lf.M)
    #plt.imshow(warped,cmap='gray')
    mid = int(img.shape[1]/2)
    leftx, lefty = lf.findLane(warped[:,:mid],leftLane)
    rightx, righty = lf.findLane(warped[:,mid:],rightLane,mid)
    rightx += mid
    
    leftLane.addLane(leftx,lefty)
    rightLane.addLane(rightx,righty)
    lf.checkDetection(leftLane,rightLane)
    leftLane.updateHistory()
    rightLane.updateHistory()
    leftLane.deriveGoodFit()
    rightLane.deriveGoodFit()
    
    processed_img = lf.drawImage(undistorted,leftLane,rightLane)
    return processed_img 
#plt.imshow(cv2.cvtColor(processed_img,cv2.COLOR_BGR2RGB))



lf = ld.LanesFinder()
leftLane = ld.Lane()
rightLane = ld.Lane()
vid_output = './output_videos/project_video3.mp4'
clip1 = VideoFileClip('./project_video.mp4')
proj_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
proj_clip.write_videofile(vid_output, audio=False)

