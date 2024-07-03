try:
    from rknnlite.api import RKNNLite as RKNN
    IsRunBoard=True
except Exception as e:
    from rknn.api import RKNN
    IsRunBoard=False

import cv2
import numpy as np
import time
import copy

def image_preprocess(image,is_normalize=False):
    """
    image:PIL，HWC
    output:NCHW，normalized between (0,1)
    """
    if not isinstance(image, np.ndarray):
        image=np.array(image)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if is_normalize:
        image=image/255.
    image_chw = np.transpose(image, (2, 0, 1))
    # Step 4: 扩展维度以形成 NCHW 格式
    image_nchw = np.expand_dims(image_chw, axis=0)
    return image_nchw

class SegRKNN(object):
    def __init__(self,model_path,IsRunBoard=True):
        self.rknn=RKNN()
        self.rknn.load_rknn(path=model_path)
        if IsRunBoard:
            self.rknn.init_runtime(
                            core_mask=RKNN.NPU_CORE_0)
        else:
            self.rknn.init_runtime(
                    target="rk3588",
                    # core_mask=RKNN.NPU_CORE_0
                    )
    def predict(self,img):
        img_new=image_preprocess(img)
        outputs = self.rknn.inference(inputs=[img_new], data_format=['nhwc'])
        return outputs
    
if __name__=="__main__":
    N=2
    model_path="checkpoints/nointerpolate_deeplabv3plus_mobilenet_ade_os8_sub.rknn"
    segrknn=SegRKNN(model_path=model_path,IsRunBoard=IsRunBoard)
    IMG_PATH="data/image.png"
    img = cv2.imread(IMG_PATH)
    resized_image = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    t1=time.time()
    for i in range(N):
        ret =segrknn.predict(img)
    t2=time.time()
    print("average time : {} s/image".format((t2-t1)/N))
    print(ret[0].shape)
        
    segrknn.rknn.release()