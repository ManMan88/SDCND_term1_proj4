import LaneDetection as ld
from moviepy.editor import VideoFileClip

def process_image(img):
    # pipeline
    #img = cv2.imread('./straight_lines1.jpg') # ADD! take image from video
    undistorted = lf.undistort(img)
    binary = lf.combinedBinary(undistorted)
    #masked_binary = lf.maskBinary(binary)
    warped = lf.prespectiveTransform(binary,lf.M)
#    warped = lf.prespectiveTransform(masked_binary,lf.M)
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
    
    processed_img = lf.drawImage(undistorted,leftLane,rightLane,mid)
    return processed_img 



lf = ld.LanesFinder()
leftLane = ld.Lane()
rightLane = ld.Lane()
vid_output = './output_videos/project_video.mp4'
clip1 = VideoFileClip('./project_video.mp4')
proj_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
proj_clip.write_videofile(vid_output, audio=False)

