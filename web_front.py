from PIL import Image
from matplotlib import pyplot as plt
from gradio import Interface, components as gr

import torch
import resnet
import torchvision


# 图像裁剪和格式转换
def process_image(img):
    # 转换成RGB格式
    img = img.convert('RGB')
    # 缩放图片
    width, height = img.size
    # 判断是否为正方形
    if width == height:
        return img
    else:
        scale = 1.0 * min(width, height) / max(width, height)
        new_width = int(500 * scale)
        new_height = int(500 * scale)
        img = img.resize((new_width, new_height))
        # 裁剪成正方形
        diff_size = abs(new_width - new_height)
        half_diff_size = diff_size // 2
        if new_width > new_height:
            crop_box = (half_diff_size, 0, half_diff_size + new_height, new_height)
        else:
            crop_box = (0, half_diff_size, new_width, half_diff_size + new_width)
        img = img.crop(crop_box)
        return img


# 制定柱形图
def get_chart(class_names, scores):
    plt.figure(figsize=(6, 5))  # 调整图像大小为竖直排布
    plt.bar(class_names, scores)  # 使用bar函数生成柱状图
    plt.xlabel("emotion")
    plt.ylabel("score")
    # 设置纵坐标长度
    plt.ylim(0, 100)
    # plt.axis('off')  # 隐藏坐标轴
    plt.tight_layout()

    # for index, value in enumerate(scores):
    #     plt.text(0, index, str(class_names[index]), fontsize=20)

    # 保存图像文件
    image_path = "vertical_bar_chart.png"
    plt.savefig(image_path)
    plt.close()

    # 读取图像文件并返回
    return image_path


# 识别图片
def predict_emotion(input_image):
    if input_image is None or not isinstance(input_image, Image.Image):
        # 如果输入为空或者不是图片，返回一个默认响应或错误信息
        return "No image provided", None
    input_image = process_image(input_image)
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize(112),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                         [0.2023, 0.1994, 0.2010])])
    # 获取模型
    clone = resnet.get_net()
    clone.load_state_dict(torch.load('resnet.params', map_location='cpu'))
    clone.eval()
    # 图像预处理
    x = transform_test(input_image)
    x = x.unsqueeze(0)  # 添加 batch 维度
    with torch.no_grad():
        y = clone(x)
    # 预测
    pre = torch.softmax(y, dim=1)
    # 表情类型
    label = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
    probability = pre.tolist()[0]
    pro = [round(i * 100, 1) for i in probability]
    emo = max(zip(probability, label))[1]
    chart = get_chart(label, pro)
    return emo, chart


# 使用Gradio界面
gradio_interface = Interface(
    fn=predict_emotion,
    title="面部表情识别",
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Predicted Expression"),  # 展示预测的类别
        gr.Image(label="Probability Distribution"),  # 展示概率分布
    ],
    live=True
)

gradio_interface.launch(server_port=7861, share=True)
