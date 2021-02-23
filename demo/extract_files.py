import os
import re
path = "/home/river/Graduation_Project/experiment/code/Python/ssd.pytorch-master/results/eval_results/"
# 获取该目录下所有文件，存入列表中
allFile = os.listdir(path)
print(len(allFile))
iteration_mAP = {}
for file in allFile:
    file_object = open(path+file)

    try:
        file_context = file_object.read()
        #  file_context = open(file).read().splitlines()
        # file_context是一个list，每行文本内容是list中的一个元素
    finally:
        file_object.close()
    pattern = re.compile(r'AP for tvmonitor = .*')
    result = pattern.findall(file_context)
    iteration = int( str(file).replace("eval_", "").replace("_iters.txt", "") )
    mAP = float( str(result).replace("['AP for tvmonitor = ", "").replace("']", "") )
    iteration_mAP[iteration] = mAP

iterms = sorted(iteration_mAP.items())
print(iterms)
arr = []

f = open("/home/river/Graduation_Project/experiment/code/Python/ssd.pytorch-master/results/tot.txt", 'a')
for x, y in iterms:
    arr.append(y)
print( "tvmonitor_y = " + str(arr).replace(",", "") + ";", file = f)
