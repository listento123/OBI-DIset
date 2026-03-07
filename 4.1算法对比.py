import os
import time
import fitz  # PyMuPDF
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import imagehash
import faiss

# ------------ 环境配置 ------------
# 确保安装以下库：
# pip install pymupdf pillow opencv-python numpy torch transformers faiss-cpu scikit-learn matplotlib

# ------------ 全局参数 ------------
DATA_DIR = "./pdf_images"  # 存储提取图像的目录
PDF_PATH = "./documents/sample.pdf"  # 示例PDF路径
QUERY_IMAGE = "./query.png"  # 查询图像路径
SAMPLE_SIZE = 100  # 实验数据集大小
RESULT_TOP_K = 5  # 检索返回结果数


# ------------ 数据准备 ------------
def extract_pdf_images(pdf_path: str, output_dir: str) -> List[str]:
    """从PDF中提取所有图像并保存为PNG"""
    doc = fitz.open(pdf_path)
    image_paths = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        img_list = page.get_images(full=True)

        for img_index, img in enumerate(img_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_path = f"{output_dir}/page{page_num}_img{img_index}.png"

            with open(image_path, "wb") as f:
                f.write(image_bytes)
            image_paths.append(image_path)

    return image_paths


# ------------ 特征提取模块 ------------
class FeatureExtractor:
    """多算法特征提取器"""

    def __init__(self):
        # 初始化CLIP模型
        from transformers import CLIPProcessor, CLIPModel
        # 替换为本地CLIP模型路径
        clip_local_path = "D:\pycharm-code\文献检索\models\clip-vit-base-patch32"
        self.clip_processor = CLIPProcessor.from_pretrained(clip_local_path)
        self.clip_model = CLIPModel.from_pretrained(clip_local_path)

        # 初始化DINOv2模型
        from transformers import AutoImageProcessor, AutoModel
        # 替换为本地DINOv2模型路径
        dino_local_path = "D:\pycharm-code\文献检索\models\dinov2-small"
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_local_path)
        self.dino_model = AutoModel.from_pretrained(dino_local_path)

    def clip_feature(self, image_path: str) -> np.ndarray:
        """CLIP特征提取"""
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
        return features.numpy().flatten()

    def dino_feature(self, image_path: str) -> np.ndarray:
        """DINOv2特征提取"""
        image = Image.open(image_path).convert("RGB")
        inputs = self.dino_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy().flatten()

    @staticmethod
    def phash(image_path: str) -> str:
        """感知哈希"""
        return str(imagehash.phash(Image.open(image_path)))


# ------------ 索引与检索模块 ------------
class VectorIndex:
    """向量索引管理"""

    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.image_paths = []

    def add_features(self, features: np.ndarray, paths: List[str]):
        """添加特征到索引"""
        self.index.add(features)
        self.image_paths.extend(paths)

    def search(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """执行相似性搜索"""
        distances, indices = self.index.search(query, k)
        return [(self.image_paths[i], distances[0][i]) for i in indices[0]]


# ------------ 评估模块 ------------
class Evaluator:
    """检索性能评估"""

    @staticmethod
    def calculate_map(query_results: Dict[str, List[str]], relevance_dict: Dict[str, List[str]]) -> float:
        """计算mAP"""
        aps = []
        for query, results in query_results.items():
            y_true = [1 if res in relevance_dict[query] else 0 for res in results]
            y_score = list(range(len(results), 0, -1))  # 简单评分：排名越高分数越高
            aps.append(average_precision_score(y_true, y_score))
        return np.mean(aps)

    @staticmethod
    def calculate_recall_at_k(query_results: Dict[str, List[str]], relevance_dict: Dict[str, List[str]],
                              k: int) -> float:
        """计算Recall@K"""
        recalls = []
        for query, results in query_results.items():
            relevant = set(relevance_dict[query])
            retrieved = set(results[:k])
            recalls.append(len(relevant & retrieved) / len(relevant))
        return np.mean(recalls)


# ------------ 主实验流程 ------------
def main_experiment():
    # 步骤1：数据准备
    print("Step 1: 准备实验数据...")
    os.makedirs(DATA_DIR, exist_ok=True)
    image_paths = extract_pdf_images(PDF_PATH, DATA_DIR)[:SAMPLE_SIZE]

    # 生成模拟相关关系（实际应用需真实标注）
    queries, database = train_test_split(image_paths, test_size=0.8)
    relevance_dict = {q: [q] + np.random.choice(database, 4).tolist() for q in queries}  # 每个查询有5个相关结果

    # 步骤2：特征提取
    print("Step 2: 提取特征...")
    extractor = FeatureExtractor()

    # 提取CLIP特征
    clip_features = np.array([extractor.clip_feature(p) for p in database])
    clip_index = VectorIndex(clip_features.shape[1])
    clip_index.add_features(clip_features, database)

    # 提取DINOv2特征
    dino_features = np.array([extractor.dino_feature(p) for p in database])
    dino_index = VectorIndex(dino_features.shape[1])
    dino_index.add_features(dino_features, database)

    # 步骤3：执行检索
    print("Step 3: 执行检索...")
    algorithms = {
        "CLIP": (extractor.clip_feature, clip_index),
        "DINOv2": (extractor.dino_feature, dino_index)
    }

    results = {}
    for algo_name, (feat_func, index) in algorithms.items():
        algo_results = {}
        for query in queries:
            query_feat = feat_func(query)
            algo_results[query] = [path for path, _ in index.search(query_feat.reshape(1, -1), RESULT_TOP_K)]
        results[algo_name] = algo_results

    # 步骤4：性能评估
    print("Step 4: 评估性能...")
    evaluation_metrics = {}
    for algo_name, algo_results in results.items():
        evaluation_metrics[algo_name] = {
            "mAP": Evaluator.calculate_map(algo_results, relevance_dict),
            "Recall@5": Evaluator.calculate_recall_at_k(algo_results, relevance_dict, 5)
        }

    # 步骤5：结果可视化
    print("\n实验结果对比:")
    print("{:<10} {:<10} {:<10}".format("Algorithm", "mAP", "Recall@5"))
    for algo, metrics in evaluation_metrics.items():
        print("{:<10} {:<10.4f} {:<10.4f}".format(algo, metrics["mAP"], metrics["Recall@5"]))

    # 绘制精度对比图
    plt.figure(figsize=(10, 6))
    x = np.arange(len(evaluation_metrics))
    width = 0.35

    plt.bar(x - width / 2, [m["mAP"] for m in evaluation_metrics.values()], width, label='mAP')
    plt.bar(x + width / 2, [m["Recall@5"] for m in evaluation_metrics.values()], width, label='Recall@5')

    plt.ylabel('Score')
    plt.title('Performance Comparison')
    plt.xticks(x, evaluation_metrics.keys())
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main_experiment()
