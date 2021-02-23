import os
import re
path = "/home/river/Graduation_Project/experiment/code/Python/ssd.pytorch-master/results/eval_results/"
# 获取该目录下所有文件，存入列表中
allFile = os.listdir(path)
print(len(allFile))
iteration_results = {}
for file in allFile:
    file_object = open(path+file)

    try:
        file_context = file_object.read()
        #  file_context = open(file).read().splitlines()
        # file_context是一个list，每行文本内容是list中的一个元素
    finally:
        file_object.close()
    # print(str(file).replace("eval_", "").replace("_iters.txt", ""))
    num = str(file).replace("eval_", "").replace("_iters.txt", "")
    iteration_results[int(num)] = file_context


f = open("/home/river/Graduation_Project/experiment/code/Python/ssd.pytorch-master/results/total_results.txt", 'w+')
iterms = sorted(iteration_results.items())
print(iterms)
arr = []
for x, y in iterms:
    arr.append( "\n" + str(x) + " iterations:\n" + y)
    arr.append("*************************************************************************"
               "********************************************************************************"
               "***************************************\n")
    arr.append("*************************************************************************"
               "********************************************************************************"
               "***************************************\n")
    arr.append("*************************************************************************"
               "********************************************************************************"
               "***************************************\n\n\n")

print(str(arr).replace("['", "").replace("']", "").replace("\\n", "\n").replace("', '", ""), file = f)
