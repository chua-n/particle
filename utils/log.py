import re
import os
import logging


def setLogger(name, output_dir="output/log/"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(os.path.join(output_dir, name+'.log'), 'w')
    c_handler.setLevel(level=logging.INFO)
    f_handler.setLevel(level=logging.DEBUG)
    format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(format)
    f_handler.setFormatter(format)
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    return logger


def parseLog(logFile, key_word):
    values = []
    with open(logFile, 'r') as log:
        for line in log:
            indBegin = line.find(key_word)
            # while循环确保没有重复的关键字
            while line[indBegin-1] != '[' or line[indBegin+len(key_word)] != ' ':
                indBegin = line.find(key_word, indBegin+len(key_word))
                if indBegin == -1:
                    break
            if indBegin == -1:
                continue
            indBegin += len(key_word)
            indEnd = line.find(']', indBegin)
            if key_word == 'D(G(z))':
                values.append(float(line[indBegin:indEnd].split('/')[-1]))
            else:
                values.append(float(line[indBegin:indEnd]))
    return values


class parseHTMLLog:
    """HTML日志信息提取处理的相关操作"""

    @staticmethod
    def get_loss(log_file):
        """从保存的HTML日志文件中提取出loss的变化情况"""

        # 从语法角度讲，末尾究竟应不应该有\s来匹配每一行最后的隐形换行符?
        mode = re.compile(
            r'Loss_re: (\d+\.\d{4}), Loss_kl: (\d+\.\d{4}), Loss: (\d+\.\d{4})\n$')
        loss_re = []
        loss_kl = []
        loss = []
        with open(log_file, 'r') as log:
            for line in log:
                target = mode.search(line)
                if target:  # 确保有匹配到对象
                    loss_re.append(float(target.group(1)))
                    loss_kl.append(float(target.group(2)))
                    loss.append(float(target.group(3)))
        return loss_re, loss_kl, loss

    @staticmethod
    def get_time(log_file):
        """从保存的HTML日志文件中提取出训练时间"""

        # mode不能强制^，因为对应html中第一行以<pre>开头，使用^会忽略掉第一行的匹配
        mode = re.compile(r'Time cost so far: (\d+)h (\d+)min (\d+)s\n$')
        time = []
        with open(log_file) as log:
            for line in log:
                target = mode.search(line)
                if target:  # 确保有匹配到对象
                    h = target.group(1)
                    m = target.group(2)
                    s = target.group(3)
                    time.append(int(h)*60 + int(m) + int(s)/60)
        return time


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    plt.style.use("bmh")

    log = "/home/chuan/soil/output/vae/lamb=10/vae.log"
    # var1 = parseLog(log, "loss")[:-27]
    var1 = parseLog(log, "loss")
    var2 = parseLog(log, "testLoss")
    var1 = np.array(var1)
    var2 = np.array(var2)
    # var1[42::43] = None
    var1[21::22] = None
    xRange1 = np.arange(len(var1))
    # xRange2 = np.arange(42, len(var1), 43)
    xRange2 = np.arange(21, len(var1), 22)

    slicer1 = slice(len(xRange1)//4, None, None)
    slicer2 = slice(len(xRange2)//4, None, None)
    # slicer1 = slice(None, None, None)
    # slicer2 = slice(None, None, None)
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.plot(xRange1[slicer1], var1[slicer1], label="train set")
    ax.plot(xRange2[slicer2], var2[slicer2], label="test set")
    ax.spines["top"].set_visible(True)
    ax.set_xlabel('iter')
    ax.set_ylabel('loss')
    # plt.legend()
    plt.savefig("/home/chuan/vaeLoss.png", dpi=150)
