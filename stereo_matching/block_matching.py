import numpy as np
import cv2
import math


class BlockMatcher:
    def __init__(self, window_size,max_disparity):
        self.window_size=window_size
        self.window_half_size=math.ceil(self.window_size/2)
        self.max_disparity=max_disparity
        self.scaler=255/max_disparity #scale factor

    def compute(self,left_image,right_image,transform_method, match_method):
        assert left_image.shape==right_image.shape
        self.height,self.width=left_image.shape
        left=self.__get_tranformer(transform_method)(left_image)
        right=self.__get_tranformer(transform_method)(right_image)
        disparity=self.__get_matcher(match_method)(left,right)

        return disparity

    def __get_tranformer(self,method):
        if method=="rank":
            return self.__rank_transform
        else:
            raise NotImplemented

    def __get_matcher(self,method):
        if method=="sse":
            return self.__sse_match
        else:
            raise NotImplemented        

    def __rank_transform(self,image):
        result=np.zeros((self.height,self.width))
        
        for i in range(self.height-self.window_size+1):
            for j in range(self.width-self.window_size+1):
                anchor=image[i+self.window_half_size,j+self.window_half_size] 
                value=(image[i:i+self.window_size,j:j+self.window_size]> anchor).sum()      
                result[i+self.window_half_size,j+self.window_half_size]=value

        return result

    def __sse_match(self,left,right):
        result=np.zeros((self.height,self.width))
        #sum of square error
        for i in range(self.height-self.window_size+1):
            for j in range(self.width-self.window_size+1):
                sse=float('inf')
                best_disparity=0
                for d in range(self.max_disparity+1):
                    if (j-d<0):
                        continue
                    left_block=left[i:i+self.window_size,j:j+self.window_size]
                    right_block=right[i:i+self.window_size,j-d:j-d+self.window_size]
                    sse_compute=((left_block-right_block)*(left_block-right_block)).sum()
                    if (sse_compute <sse):
                        sse=sse_compute
                        best_disparity=d
                result[i+self.window_half_size,j+self.window_half_size]=best_disparity*self.scaler

        return result





if __name__== "__main__":
    left=cv2.imread("images/0000000000-L.png",0)
    right=cv2.imread("images/0000000000-R.png",0)
    matcher=BlockMatcher(21,100)

    disparity=matcher.compute(left,right,transform_method="rank",match_method="sse")
    disparity=cv2.medianBlur(disparity.astype("uint8"),7)

    cv2.imwrite("results/disparity.png",disparity)