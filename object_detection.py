import os
import cv2
import random
from tqdm import tqdm
from ultralytics import YOLO

# Словарь с названиями классов объектов
names_class = {
    0: "человек",
    1: "велосипед",
    2: "автомобиль",
    3: "мотоцикл",
    4: "самолет",
    5: "автобус",
    6: "поезд",
    7: "грузовик"
}

# Словарь для классов с их цветами
class_colors = {
    0: (0, 255, 0),      # Человек - зелёный
    1: (0, 0, 255),      # Велосипед - красный
    2: (255, 0, 0),      # Автомобиль - синий
    3: (255, 255, 0),    # Мотоцикл - жёлтый
    4: (0, 255, 255),    # Самолет - бирюзовый
    5: (255, 0, 255),    # Автобус - фиолетовый
    6: (128, 128, 0),    # Поезд - оливковый
    7: (128, 0, 128)     # Грузовик - пурпурный
}

def load_model(weights_path):
    """
    Загружает модель YOLO по указанному пути к весам.

    :param weights_path: Путь к файлу с весами модели YOLO.
    :return: Загруженная модель YOLO.
    """
    model = YOLO(weights_path)
    return model

def detect_objects(image, model, selected_classes):
    """
    Выполняет обнаружение объектов на изображении с использованием модели YOLO.

    :param image: Изображение для анализа.
    :param model: Загруженная модель YOLO.
    :param selected_classes: Список классов для обнаружения.
    :return: Список детектированных объектов (класс, уверенность модели, координаты рамки).
    """
    results = model(image)
    detections = []
    for result in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = result
        if int(cls) in selected_classes:
            detections.append((int(cls), conf, (int(x1), int(y1), int(x2), int(y2))))
    return detections

def draw_boxes(image, detections):
    """
    Рисует рамки и метки уверенности на изображении.

    :param image: Изображение, на котором будут рисоваться bbox (рамки).
    :param detections: Список детектированных объектов.
    :return: Изображение с нарисованными bbox (рамками).
    """
    for cls, conf, (x1, y1, x2, y2) in detections:
        label = f"{conf:.2f}"
        color = class_colors.get(cls, (0, 255, 0))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def save_frame(frame, detections, output_folder, frame_index, draw_boxes_flag, class_names):
    """
    Сохраняет кадр с детектированными объектами в папке, разделенной по классам.

    :param frame: Кадр для сохранения.
    :param detections: Список детектированных объектов на кадре.
    :param output_folder: Папка для сохранения результатов.
    :param frame_index: Индекс кадра для имени файла.
    :param draw_boxes_flag: Флаг для рисования рамок.
    :param class_names: Словарь с именами классов.
    """
    frame_with_boxes_all = frame.copy()
    class_detections = {}
    for cls, conf, bbox in detections:
        if cls not in class_detections:
            class_detections[cls] = []
        class_detections[cls].append((cls, conf, bbox))

    for cls, detections_for_class in class_detections.items():
        class_folder = os.path.join(output_folder, class_names[cls])
        os.makedirs(class_folder, exist_ok=True)

        frame_with_boxes_class = frame.copy()
        if draw_boxes_flag:
            frame_with_boxes_class = draw_boxes(frame_with_boxes_class, detections_for_class)

        output_path_class = os.path.join(class_folder, f"frame_{frame_index:04d}.jpg")
        cv2.imwrite(output_path_class, frame_with_boxes_class)

    if draw_boxes_flag:
        frame_with_boxes_all = draw_boxes(frame_with_boxes_all, detections)

    output_path_all = os.path.join(output_folder, f"frame_{frame_index:04d}_with_boxes.jpg")
    cv2.imwrite(output_path_all, frame_with_boxes_all)

def extract_frames(video_path, output_folder, model_weights, selected_classes, mode, frame_count,
                   resize_width, resize_height, interval_start, interval_end, draw_boxes_flag):
    """
    Извлекает кадры из видео, выполняет обнаружение объектов и сохраняет результаты.

    :param video_path: Путь к видеофайлу.
    :param output_folder: Папка для сохранения кадров.
    :param model_weights: Путь к весам модели YOLO.
    :param selected_classes: Список классов для обнаружения.
    :param mode: Режим выборки кадров ('random' или 'interval').
    :param frame_count: Количество кадров для извлечения.
    :param resize_width: Ширина для изменения размера кадров (0 для пропуска).
    :param resize_height: Высота для изменения размера кадров (0 для пропуска).
    :param interval_start: Начало интервала (в секундах), если режим 'interval'.
    :param interval_end: Конец интервала (в секундах), если режим 'interval'.
    :param draw_boxes_flag: Флаг для рисования рамок на кадрах.
    """
    model = load_model(model_weights)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if interval_start and interval_end:
        interval_start = max(0, int(interval_start * fps))
        interval_end = min(total_frames, int(interval_end * fps))

    if mode == 'random':
        frame_indices = sorted(random.sample(range(total_frames), frame_count))
    else:
        frame_indices = list(
            range(interval_start, interval_end, max(1, (interval_end - interval_start) // frame_count)))

    os.makedirs(output_folder, exist_ok=True)
    report_path = os.path.join(output_folder, "report.txt")
    with open(report_path, "w") as report:
        report.write("Отчет о кадрах:\n")

    for frame_index in tqdm(frame_indices, desc="Обработка кадров"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            print(f"Не удалось загрузить кадр {frame_index}.")
            continue

        if resize_width > 0 and resize_height > 0:
            frame = cv2.resize(frame, (resize_width, resize_height))

        detections = detect_objects(frame, model, selected_classes)
        save_frame(frame, detections, output_folder, frame_index, draw_boxes_flag, names_class)

        with open(report_path, "a") as report:
            report.write(f"Кадр {frame_index}: {len(detections)} детект(-ов)\n")

    cap.release()
    print(f"Отчет сохранен: {report_path}")

def get_classes():
    """
    Запрашивает у пользователя выбор классов объектов для обнаружения.

    :return: Список выбранных классов.
    """
    print("Доступные классы:")
    for key, value in names_class.items():
        print(f"{key} - {value}")

    selected_classes = []
    while True:
        try:
            classes_input = input("Введите классы для обнаружения (через запятую): ")
            classes = list(map(int, classes_input.split(',')))
            if all(cls in names_class for cls in classes):
                selected_classes = classes
                break
            else:
                print("Некорректный класс. Пожалуйста, введите существующие классы.")
        except ValueError:
            print("Некорректный ввод. Пожалуйста, введите числа.")

    return selected_classes

def get_valid_input(prompt, valid_values=None, value_type=str):
    """
    Запрашивает у пользователя ввод, проверяя корректность введенных данных.

    :param prompt: Сообщение для запроса.
    :param valid_values: Список допустимых значений.
    :param value_type: Тип значения (str, int, float).
    :return: Корректный ввод пользователя.
    """
    while True:
        user_input = input(prompt).strip()
        if value_type == int:
            try:
                user_input = int(user_input)
                return user_input
            except ValueError:
                print("Некорректный ввод. Пожалуйста, введите целое число.")
        elif value_type == float:
            try:
                user_input = float(user_input)
                return user_input
            except ValueError:
                print("Некорректный ввод. Пожалуйста, введите число.")
        elif valid_values and user_input.lower() in valid_values:
            return user_input.lower()
        else:
            print(f"Некорректный ввод. Пожалуйста, введите одно из: {', '.join(valid_values)}")

def main():
    """
    Главная функция, которая инициирует процесс извлечения кадров из видео с детекцией объектов.
    """
    video_path = input("Введите путь к видеофайлу: ").strip('\"\'')
    output_folder = input("Введите путь к папке для сохранения кадров: ").strip('\"\'')
    default_weights = "\weights\yolo11n.pt"
    print(f"Веса YOLO по умолчанию: {default_weights}")
    model_weights = input(
        "Введите путь к файлу весов модели (или нажмите Enter для использования по умолчанию): ").strip('\"\'')
    if not model_weights:
        model_weights = default_weights

    selected_classes = get_classes()

    mode = get_valid_input("Выберите режим (random/interval): ", ["random", "interval"])
    if mode == "interval":
        interval_start = get_valid_input("Введите начало интервала (в секундах): ", value_type=float)
        interval_end = get_valid_input("Введите конец интервала (в секундах): ", value_type=float)
    else:
        interval_start = interval_end = None

    frame_count = get_valid_input("Введите количество кадров для извлечения: ", value_type=int)
    resize_width = get_valid_input("Введите ширину для изменения размера (0 для пропуска): ", value_type=int)
    resize_height = get_valid_input("Введите высоту для изменения размера (0 для пропуска): ", value_type=int)
    draw_boxes_flag = get_valid_input("Рисовать рамки на кадрах? (y/n): ", valid_values=["y", "n"]) == "y"

    extract_frames(video_path, output_folder, model_weights, selected_classes, mode, frame_count, resize_width,
                   resize_height, interval_start, interval_end, draw_boxes_flag)

if __name__ == "__main__":
    main()
