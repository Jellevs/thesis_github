import cv2

class TeamAssigner2:
    def __init__(self):
        pass

    def _crop_top_half(self, frame, bbox):
        img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_img = img[0: int(img.shape[0]/2), :]

        return top_half_img
    

    def _siglip_embed(self,  bgr_crop):
        

