import time
import onnx
import numpy as np
import onnxruntime as ort


start = time.time()
print("load model ....")
model = onnx.load_model(r'D:\code\python\PycharmProjects\onnx_deploy_python\qr_seg_df_200_0217.onnx')
print("load model success!!!: {}".format(time.time()-start))

start = time.time()
print("open InferenceSession ....")
ort_session = ort.InferenceSession(r'D:\code\python\PycharmProjects\onnx_deploy_python\qr_seg_df_200_0217.onnx')
print("open InferenceSession success!!! time: {}".format(time.time() - start))

while 1:
    start = time.time()
    outputs = ort_session.run(None, {'input.1': np.random.randn(1, 3, 512, 512).astype(np.float32)})  # inputs is same to orig
    print("run one image time: {}".format(time.time() - start))



