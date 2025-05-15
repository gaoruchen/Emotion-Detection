import sys
import torch
import torchvision

from PIL import Image
from PyQt5 import QtCore, QtWidgets
from matplotlib import pyplot as plt
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap

from gui import Ui_MainWindow
import resnet


def get_chart(class_names, scores):
    plt.figure(figsize=(7, 10))  # 调整图像大小为竖直排布
    plt.barh(class_names, scores)  # 使用barh函数生成水平柱状图
    plt.xlabel("emotion")
    plt.ylabel("pro")
    plt.axis('off')  # 隐藏坐标轴
    plt.tight_layout()

    for index, value in enumerate(scores):
        plt.text(0, index, str(class_names[index]), fontsize=20)

    # 保存图像文件
    image_path = "vertical_bar_chart.png"
    plt.savefig(image_path)
    plt.close()

    # 读取图像文件并返回
    return image_path


class MyWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.init_ui()
        # 测试图片定义
        self.selected_image = None

    def init_ui(self):
        # 事件定义
        self.openfile.clicked.connect(self.get_image)
        self.recognize.clicked.connect(self.test_image)

    # 槽函数
    # 打开图片
    def get_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp *.gif)",
                                                   options=options)
        if file_name:
            self.selected_image = Image.open(file_name)
            pixmap = QPixmap(file_name)
            self.image.setPixmap(pixmap)
            self.image.setScaledContents(True)  # 图片自适应大小
            self.image.show()

    # 识别图片函数
    def test_image(self):
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.Resize(112),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                             [0.2023, 0.1994, 0.2010])])

        clone = resnet.get_net()
        clone.load_state_dict(torch.load('resnet.params', map_location='cpu'))
        clone.eval()
        # # folder_path = 'data/train/Happiness'
        # files = os.listdir(folder_path)
        # image = Image.open(os.path.join(folder_path, files[0]))
        x = transform_test(self.selected_image)
        x = x.unsqueeze(0)  # 添加 batch 维度
        with torch.no_grad():
            y = clone(x)
        pre = torch.softmax(y, dim=1)
        # 显示表情类型
        label = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
        probability = pre.tolist()[0]
        pro = [round(i*100, 1) for i in probability]
        emo = max(zip(probability, label))[1]
        self.result_emo.setPlainText(emo)
        self.result_emo.setStyleSheet("font-size: 30px;")
        self.result_emo.show()
        # 显示图像
        pixmap = QPixmap(get_chart(label, pro))
        self.chart.setPixmap(pixmap)
        self.chart.setScaledContents(True)  # 图片自适应大小
        self.chart.show()
        # print(emo)
        # print(probability)
        # print(pro)


QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
app = QtWidgets.QApplication(sys.argv)
window = QtWidgets.QDialog()
mainmenu = MyWindow()
mainmenu.show()
sys.exit(app.exec_())
