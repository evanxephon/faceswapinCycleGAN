import cv2
import os
import sys
sys.path.append('..')
from src import detect_faces
from PIL import Image,ImageOps,ImageDraw
#import matplotlib.pyplot as plt 
from umeyama import umeyama
import numpy as np

def getFaceInVideo(videoPath, svPath, eyemaskon=False,fps=10):
    
    if not os.path.isdir(svPath):
        os.mkdir(svPath)
    if not os.path.isdir(svPath + 'rgb/'):
        os.mkdir(svPath + 'rgb/')
    
    cap = cv2.VideoCapture(videoPath)
    
    numFrame = 0
    
    while True:
        
        if cap.grab():
            
            flag, frame = cap.retrieve()
            numFrame += 1
            if numFrame % fps == 0:
                
                # frame matrix is BGR mode, needs to turn it into RGB mode
                frame = frame[:,:,::-1]
                frame = Image.fromarray(frame,mode='RGB')
                
                bounding_boxes, landmarks = detect_faces(frame)
                
                # using the bounding_boxes to cut the face
                face = frame.crop(bounding_boxes.flatten()[:4])
                # get eyemask 
                if eyemaskon:
                    eyemask = getEyeMask(np.zeros(face.shape[:2]), landmarks[:2,:])
                    eyemask = eyemask[:,:,None]
                
                src_landmarks = get_src_landmarks(bounding_boxes.flatten()[:4], landmarks.flatten())
                #print(f'srclandmark:{src_landmarks}')
                
                tar_landmarks = get_tar_landmarks(face)
                #print(f'tarlandmark:{tar_landmarks}')
                
                align_face = landmarks_match_mtcnn(np.array(face)[:,:,::-1], src_landmarks, tar_landmarks)
                align_mask_eye = landmark_match_mtcnn(eyemask, src_landmarks, tar_landmarks)
                
                facePath = svPath + str(numFrame//fps) + ".jpg"
                face.save(facePath,'JPEG')
                
                alignFacePath = svPath + 'rgb/' + str(numFrame//fps) + ".jpg"
                alignEyeMaskPath = svPath + 'eyemask/' + str(numFrame//fps) + ".jpg"
                #align_face.save(alignFacePath,'JPEG')
                cv2.imwrite(alignFacePath, align_face)     
                cv2.imwrite(alignEyeMaskPath, align_mask_eye)
        else:
            break
            
# emm, now i know we just use [::-1] to reverse channel           
def BGRtoRGB(frame):
    for row in range(len(frame)):
        for i in range(len(frame[row])):
            channel = []
            channel.append(frame[row][i][0])
            channel.append(frame[row][i][1])
            channel.append(frame[row][i][2])
            #print(tunnel)
            frame[row][i][0] = tunnel[2]
            frame[row][i][1] = tunnel[1]
            frame[row][i][2] = tunnel[0]
            
    return frame
            
def getFaceInImages(imagePath, svpath):
    # traversal the path for the pic
    number = 0
    for filepath, _, files in os.walk(imagePath):
        for file in files:
            
            image = Image.open(os.path.join(filepath,file))
                               
            bounding_boxes, landmarks = detect_faces(frame)
                               
            face = image.crop(bounding_boxes.flatten()[:4])
                               
            newPath = svPath + f'{number}' + ".jpg"
                               
            face.save(newPath,'JPEG')
                               
            number += 1
            
            
def get_src_landmarks(bbox, landmarks):
    """
    bbox : boundbox coord.
    landmarks: landmarks predicted by MTCNN
    get the relatively position
    """    
    src_landmarks = [(int(landmarks[i]-bbox[0]), 
                      int(landmarks[i+5]-bbox[1])) for i in range(5)]
    
    return src_landmarks


def get_tar_landmarks(img):
    """    
    img: detected face image
    get the aligned face's landmarks
    """         
    ratio_landmarks = [
        (0.31339227236234224, 0.3259269274198092),
        (0.31075140146108776, 0.7228453709528997),
        (0.5523683107816256, 0.5187296867370605),
        (0.7752419985257663, 0.37262483743520886),
        (0.7759613623985877, 0.6772957581740159)
        ]   
        
    img_size = np.array(img).shape
    tar_landmarks = [(int(xy[1]*img_size[1]), 
                      int(xy[0]*img_size[0])) for xy in ratio_landmarks]
    
    return tar_landmarks

def landmarks_match_mtcnn(src_im, src_landmarks, tar_landmarks): 
    """
    umeyama(src, dst, estimate_scale)
    landmarks coord. for umeyama should be (width, height) or (y, x)
    """
    src_size = src_im.shape
    #print(f'src_size:{src_size}')
    
    src_tmp = [(int(xy[0]), int(xy[1])) for xy in src_landmarks]
    tar_tmp = [(int(xy[0]), int(xy[1])) for xy in tar_landmarks]
    
    mat = umeyama(np.array(src_tmp), np.array(tar_tmp), False)[0:2]
    result = cv2.warpAffine(src_im, mat, (src_size[1], src_size[0]), borderMode=cv2.BORDER_REPLICATE) 
    
    #print(f'result:{result}')
    #print(f'resultsize:{result.shape}')
    return result

def getEyeMask(face, landmarks):
    h = face.shape[0]
    w = face.shape[1]
    kernel = np.ones([h//12, w//10], np.uint8)
    face[landmarks[0][0],landmarks[0][1],:] = 255
    face[landmarks[1][0],landmarks[1][1],:] = 255
    eyemask = cv2.dilate(face, kernel, iteration=1)
    return eyemask
    
if __name__ == '__main__':
    os.chdir('..')
    getFaceInVideo('./bush.avi','./faceA/',eyemaskon=False, fps=10)
    getFaceInVideo('./putin.avi','./faceB/',eyemaskon=False, fps=10)
