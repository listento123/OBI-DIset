import fitz  # PyMuPDF
import os


def extract_images_from_pdf(pdf_path, output_folder):
    # 打开PDF文件
    doc = fitz.open(pdf_path)

    # 检查并创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 初始化图像计数
    image_count = 0

    # 遍历每一页并提取图像
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)  # 加载当前页
        image_list = page.get_images(full=True)  # 获取页面中的图像列表

        # 如果当前页没有图像，跳过
        if not image_list:
            print(f"第{page_num + 1}页没有图像")
            continue

        # 遍历图像列表并保存图像
        for img_index, image in enumerate(image_list):
            xref = image[0]  # 获取图像的xref
            base_image = doc.extract_image(xref)  # 提取图像
            image_bytes = base_image["image"]  # 获取图像的字节数据
            image_extension = base_image["ext"]  # 获取图像格式（如png、jpeg）

            # 设置图像保存路径
            image_path = os.path.join(output_folder, f"page_{page_num + 1}_img_{img_index + 1}.{image_extension}")

            # 保存图像
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)

            print(f"提取并保存了图像: {image_path}")
            image_count += 1

    print(f"总共提取了 {image_count} 张图像。")


# 设置PDF文件路径和图像保存文件夹路径
pdf_path = r"D:\pycharm-code\文献检索\long.pdf"  # 替换为你的PDF文件路径
output_folder = "extracted_images"  # 输出文件夹

# 调用函数提取图像
extract_images_from_pdf(pdf_path, output_folder)
