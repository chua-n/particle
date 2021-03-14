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
            if indBegin == -1:
                continue
            indBegin += len(key_word)
            indEnd = line.find(']', indBegin)
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

    log = "/home/chuan/soil/output/vae/vae.log"
    var1 = parseLog(log, "loss_re")
    var2 = parseLog(log, "testLoss_re")
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(16, 12))
    ax[0].plot(var1)
    ax[0].set(ylabel="loss_re")
    ax[1].plot(var2)
    ax[1].set(ylabel="testLoss_re")
    plt.xlabel('iter')
    plt.show()
