import os
import json
import yaml
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from tqdm import tqdm

def load_config(student_id):
    """
    Загружает конфигурацию для вашего ID студента
    """
    config_path = f"configs/student_{student_id}.json"
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Загружена конфигурация для студента {student_id}")
        print(f"Класс: {config['target_class']}")
        print(f"Архитектура: {config['arch']}")
        return config
    except FileNotFoundError:
        print(f"Ошибка: Файл конфигурации {config_path} не найден.")
        print(f"Убедитесь, что вы правильно указали student_id = {student_id}")
        return None

def check_dataset_exists(student_id):
    """
    Проверяет наличие предварительно подготовленного датасета
    """
    dataset_dir = f"student_{student_id}"
    if not os.path.exists(dataset_dir):
        print(f"Ошибка: Директория датасета {dataset_dir} не найдена.")
        print("Убедитесь, что вы скачали и распаковали архив с датасетом в корневую директорию.")
        return False
    return True

def load_dataset_config(student_id):
    """
    Загружает конфигурацию датасета из файла data.yaml
    """
    yaml_path = f"student_{student_id}/data.yaml"
    if not os.path.exists(yaml_path):
        print(f"Ошибка: Файл конфигурации датасета {yaml_path} не найден.")
        return None
    
    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    return data_config

def load_dataset_to_df(student_id, split='train'):
    """
    Загружает изображения и метки в pandas DataFrame
    """
    dataset_dir = f"student_{student_id}"
    images_dir = os.path.join(dataset_dir, split, "images")
    labels_dir = os.path.join(dataset_dir, split, "labels")
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"Ошибка: Директории {split} данных не найдены.")
        return None
    
    # Получаем список файлов изображений
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    data = []
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt')
        
        if os.path.exists(label_path):
            # Загружаем изображение для получения размеров
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            height, width = img.shape[:2]
            
            # Загружаем метки
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:  # формат YOLO: class x_center y_center width height
                        class_id = int(parts[0])
                        # Нормализованные координаты
                        x_center, y_center, bbox_width, bbox_height = map(float, parts[1:5])
                        
                        # Сохраняем данные
                        data.append({
                            'image_path': img_path,
                            'width': width,
                            'height': height,
                            'bbox': [x_center, y_center, bbox_width, bbox_height],
                            'class_id': class_id
                        })
                        break  # Берем только первый bbox для каждого изображения
    
    # Создаем DataFrame
    df = pd.DataFrame(data)
    print(f"Загружено {len(df)} примеров для {split} набора")
    
    # Выводим пример данных
    if len(df) > 0:
        print("\nПример данных:")
        sample = df.iloc[0]
        print(f"Путь к изображению: {sample['image_path']}")
        print(f"Размеры изображения: {sample['width']}x{sample['height']}")
        print(f"Координаты bbox (x_center, y_center, width, height): {sample['bbox']}")
        
        # Визуализируем пример
        img = cv2.imread(sample['image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        visualize_prediction(img, sample['bbox'], sample['bbox'], 1.0)
    
    return df

class ObjectDetectionDataset(Dataset):
    def __init__(self, data_df, transform=None):
        """
        Класс датасета для задачи детекции объектов
        
        Args:
            data_df (pd.DataFrame): DataFrame с данными
            transform: Преобразования для изображений
        """
        self.data = data_df
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Загрузка и преобразование изображения и bbox
        """
        # Получаем путь к изображению и bbox
        row = self.data.iloc[idx]
        img_path = row['image_path']
        bbox = row['bbox']  # [x_center, y_center, width, height]
        
        # Загружаем изображение
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Преобразуем bbox в тензор
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)
        
        # Преобразуем изображение в тензор, если еще не преобразовано
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        return image, bbox_tensor

def set_seed(seed):
    """
    Устанавливает seed для воспроизводимости результатов
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def bbox_iou(box1, box2):
    """
    Вычисляет IoU (Intersection over Union) между двумя bbox
    Формат bbox: [x_center, y_center, width, height] (нормализованный [0-1])
    """
    # Конвертируем формат центр-ширина-высота в координаты углов
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2
    
    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2
    
    # Вычисляем координаты пересечения
    intersect_x1 = max(box1_x1, box2_x1)
    intersect_y1 = max(box1_y1, box2_y1)
    intersect_x2 = min(box1_x2, box2_x2)
    intersect_y2 = min(box1_y2, box2_y2)
    
    # Вычисляем площадь пересечения
    intersect_width = max(0, intersect_x2 - intersect_x1)
    intersect_height = max(0, intersect_y2 - intersect_y1)
    intersection = intersect_width * intersect_height
    
    # Вычисляем площади bbox
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    
    # Вычисляем IoU
    union = box1_area + box2_area - intersection
    
    # Избегаем деления на ноль
    if union == 0:
        return 0
    
    return intersection / union

def visualize_prediction(image, true_bbox, pred_bbox, iou, save_path=None):
    """
    Визуализирует изображение с истинным и предсказанным bbox
    """
    # Преобразуем тензор в массив numpy, если нужно
    if isinstance(image, torch.Tensor):
        if image.shape[0] == 3:  # Если формат CxHxW, преобразуем в HxWxC
            image = image.permute(1, 2, 0)
        image = image.cpu().numpy()
    
    # Обработка нормализованных изображений
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    if image.shape[2] == 3:  # RGB
        pass
    elif image.shape[2] == 1:  # Grayscale
        image = np.squeeze(image)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    
    # Получаем размеры изображения
    height, width = image.shape[:2]
    
    # Преобразуем нормализованные координаты в абсолютные
    true_x1 = int((true_bbox[0] - true_bbox[2]/2) * width)
    true_y1 = int((true_bbox[1] - true_bbox[3]/2) * height)
    true_w = int(true_bbox[2] * width)
    true_h = int(true_bbox[3] * height)
    
    pred_x1 = int((pred_bbox[0] - pred_bbox[2]/2) * width)
    pred_y1 = int((pred_bbox[1] - pred_bbox[3]/2) * height)
    pred_w = int(pred_bbox[2] * width)
    pred_h = int(pred_bbox[3] * height)
    
    # Рисуем истинный bbox (зеленый)
    true_rect = patches.Rectangle(
        (true_x1, true_y1), true_w, true_h, 
        linewidth=2, edgecolor='g', facecolor='none', label='GT')
    ax.add_patch(true_rect)
    
    # Рисуем предсказанный bbox (красный)
    pred_rect = patches.Rectangle(
        (pred_x1, pred_y1), pred_w, pred_h, 
        linewidth=2, edgecolor='r', facecolor='none', label='Pred')
    ax.add_patch(pred_rect)
    
    # Добавляем информацию об IoU
    ax.set_title(f'IoU: {iou:.4f}')
    ax.legend()
    
    # Убираем оси
    ax.axis('off')
    
    # Сохраняем или показываем
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def create_validation_dataset(student_id, transform=None):
    """
    Создает простой датасет для валидации модели
    """
    dataset_dir = f"student_{student_id}"
    valid_images_dir = os.path.join(dataset_dir, "valid", "images")
    valid_labels_dir = os.path.join(dataset_dir, "valid", "labels")
    
    if not os.path.exists(valid_images_dir) or not os.path.exists(valid_labels_dir):
        raise FileNotFoundError(
            f"Директории валидационных данных не найдены. Убедитесь, что вы распаковали датасет в корневую директорию проекта."
        )
    
    # Получаем список файлов
    image_files = [os.path.join(valid_images_dir, f) for f in os.listdir(valid_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Для каждого изображения находим соответствующий файл с метками
    valid_data = []
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(valid_labels_dir, label_name)
        
        if os.path.exists(label_path):
            # Загружаем изображение для получения размеров
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            h, w = img.shape[:2]
            
            # Загружаем метки
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    # YOLO формат: class_id x_center y_center width height
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        # Нормализованные координаты
                        bbox = [float(p) for p in parts[1:]]
                        valid_data.append((img_path, bbox))
                        break  # Берем только первый объект
    
    class ValidationDataset(Dataset):
        def __init__(self, data, transform=None):
            self.data = data
            self.transform = transform or transforms.Compose([
                transforms.ToTensor(),
            ])
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            img_path, bbox = self.data[idx]
            
            # Загружаем изображение
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Применяем трансформации
            if self.transform:
                img = self.transform(img)
            
            bbox = torch.tensor(bbox, dtype=torch.float32)
            
            return img, bbox, img_path
    
    return ValidationDataset(valid_data, transform)

def autocheck_model(model, student_id, device=None, num_display=3, threshold=0.5, batch_size=8, transform=None):
    """
    Автоматически проверяет модель на валидационном наборе
    и возвращает метрики качества, а также визуализирует результаты
    
    Args:
        model: модель для оценки
        student_id: ID студента
        device: устройство для вычислений (cpu/cuda)
        num_display: количество изображений для отображения
        threshold: порог IoU для успешного предсказания
        batch_size: размер батча для валидации
        transform: трансформации для применения к изображениям
    
    Returns:
        mean_iou: средний IoU по всему валидационному набору
        success_rate: процент предсказаний с IoU >= threshold
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Создаем датасет для валидации
    valid_dataset = create_validation_dataset(student_id, transform)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    # Переводим модель в режим оценки и на нужное устройство
    model.eval()
    model.to(device)
    
    all_ious = []
    sample_results = []
    
    with torch.no_grad():
        for batch_idx, (images, targets, img_paths) in enumerate(tqdm(valid_loader, desc="Validating")):
            images = images.to(device)
            targets = targets.to(device)
            
            # Получаем предсказания модели
            predictions = model(images)
            
            # Вычисляем IoU для каждого изображения в батче
            for i, (pred, target) in enumerate(zip(predictions, targets)):
                # Приводим предсказания к нужному формату
                pred = pred.cpu().numpy()
                target = target.cpu().numpy()
                
                # Убеждаемся, что координаты ограничены [0, 1]
                pred = np.clip(pred, 0, 1)
                
                # Вычисляем IoU
                iou = bbox_iou(pred, target)
                all_ious.append(iou)
                
                # Сохраняем информацию о некоторых примерах для визуализации
                if len(sample_results) < num_display:
                    sample_results.append((
                        images[i].cpu(),
                        target,
                        pred,
                        iou,
                        img_paths[i]
                    ))
    
    # Вычисляем метрики
    mean_iou = np.mean(all_ious)
    success_rate = np.mean([iou >= threshold for iou in all_ious]) * 100
    
    # Выводим результаты
    print(f"\nРезультаты оценки модели:")
    print(f"Средний IoU: {mean_iou:.4f}")
    print(f"Успешных предсказаний (IoU >= {threshold}): {success_rate:.2f}%")
    
    # Определяем итоговый статус
    if mean_iou >= threshold and success_rate >= 75:
        print("\n✅ ЗАДАНИЕ ВЫПОЛНЕНО УСПЕШНО! ✅")
    else:
        print("\n❌ Задание не выполнено. Необходимо улучшить модель. ❌")
        if mean_iou < threshold:
            print(f"Требуется средний IoU >= {threshold}, получено {mean_iou:.4f}")
        if success_rate < 75:
            print(f"Требуется успешность >= 75%, получено {success_rate:.2f}%")
    
    # Визуализируем несколько примеров
    print("\nПримеры предсказаний:")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    for i, (image, target, pred, iou, img_path) in enumerate(sample_results):
        print(f"Пример {i+1}: IoU = {iou:.4f}, Результат: {'✅' if iou >= threshold else '❌'}")
        save_path = os.path.join(results_dir, f"example_{i+1}.png")
        visualize_prediction(image, target, pred, iou, save_path)
        print(f"Визуализация сохранена в {save_path}")
    
    # Возвращаем метрики
    return mean_iou, success_rate