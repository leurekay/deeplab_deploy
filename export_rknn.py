import cv2
import numpy as np
from rknn.api import RKNN
import time






def image_preprocess(image,is_normalize=True):
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

if __name__=="__main__":
    N=2
    #mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]
    IMG_PATH="data/image.png"
    src_model_path="checkpoints/best_deeplabv3plus_mobilenet_ade_os8.onnx"

    rknn=RKNN(verbose=True,verbose_file='checkpoints/to_rknn.log')
    rknn.config(mean_values=[[0.485, 0.456, 0.406]],
            std_values=[[0.229, 0.224, 0.225]],
                target_platform="rk3588")
    # rknn.load_pytorch(model="/data/cv/object_track/models/yolov8n.pt",
    #                     input_size_list=[[1,3,224,224]])
    # rknn.load_onnx(model="/data/cv/rknn_deploy/rknn-toolkit2/rknn-toolkit2/examples/onnx/yolov5/yolov5s_relu.onnx")
    rknn.load_onnx(model=src_model_path)
    # rknn.build(do_quantization=True,
    #            dataset="images/imglist.txt")
    rknn.export_rknn("{}.rknn".format(src_model_path.split(".")[0]))

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target="rk3588",
                            target_sub_class=None,
                            device_id=None,
                            perf_debug=True,
                            # core_mask=RKNN.NPU_CORE_0_1_2,
                            eval_mem=True
                            )
    # rknn.eval_perf(inputs=[IMG_PATH],
    #                data_format=None,
    #                is_print=True)
    # rknn.eval_memory(is_print=True)
    # if ret != 0:
    #     print('Init runtime environment failed!')
    #     exit(ret)
    # print('done')   

        # Set inputs
    
    img = cv2.imread(IMG_PATH)
    img_new=image_preprocess(img)

    # Inference
    print('--> Running model')
    t1=time.time()
    for _ in range(N):
        outputs = rknn.inference(inputs=[img_new], data_format=['nchw'])
    t2=time.time()
    print("average time : {}".format((t2-t1)/N))
    for output in outputs:
        print(output.shape)
        
    rknn.release()