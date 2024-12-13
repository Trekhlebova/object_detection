import numpy as np
from unittest.mock import MagicMock, patch
import os
import pytest
from object_detection import detect_objects, save_frame, load_model, draw_boxes, names_class

@patch('object_detection.YOLO')
def test_detect_objects_positive(MockYOLO):
    mock_model = MagicMock()
    MockYOLO.return_value = mock_model
    mock_result = [
        MagicMock(boxes=MagicMock(data=np.array([[0, 0, 10, 10, 0.9, 2]])))
    ]
    mock_model.return_value = mock_result

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    selected_classes = [2, 3]
    detections = detect_objects(image, mock_model, selected_classes)

    assert len(detections) == 1
    assert detections[0] == (2, 0.9, (0, 0, 10, 10))

@patch('object_detection.YOLO')
def test_detect_objects_negative(MockYOLO):
    mock_model = MagicMock()
    MockYOLO.return_value = mock_model
    mock_result = [
        MagicMock(boxes=MagicMock(data=np.array([])))
    ]
    mock_model.return_value = mock_result

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    selected_classes = [2, 3]
    detections = detect_objects(image, mock_model, selected_classes)

    assert len(detections) == 0

@patch('os.makedirs')
@patch('cv2.imwrite')
def test_save_frame_positive(mock_imwrite, mock_makedirs):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [(2, 0.9, (10, 10, 50, 50))]
    output_folder = 'output'
    frame_index = 1
    draw_boxes_flag = True

    save_frame(frame, detections, output_folder, frame_index, draw_boxes_flag, names_class)

    mock_makedirs.assert_called_once_with(os.path.join(output_folder, names_class[2]), exist_ok=True)
    mock_imwrite.assert_called()

@patch('os.makedirs')
@patch('cv2.imwrite')
def test_save_frame_negative(mock_imwrite, mock_makedirs):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [(2, 0.9, (10, 10, 50, 50))]
    output_folder = 'output'
    frame_index = 1
    draw_boxes_flag = True

    mock_imwrite.side_effect = Exception("Error saving file")
    with pytest.raises(Exception, match="Error saving file"):
        save_frame(frame, detections, output_folder, frame_index, draw_boxes_flag, names_class)

@patch('object_detection.YOLO')
def test_load_model_positive(MockYOLO):
    mock_model = MagicMock()
    MockYOLO.return_value = mock_model

    model = load_model('weights/yolo11n.pt')
    assert model == mock_model

@patch('object_detection.YOLO')
def test_load_model_negative(MockYOLO):
    MockYOLO.side_effect = Exception("Model file not found")
    with pytest.raises(Exception, match="Model file not found"):
        load_model('weights/nonexistent.pt')

def test_names_class_positive():
    assert isinstance(names_class, dict)
    assert all(isinstance(k, int) for k in names_class.keys())
    assert all(isinstance(v, str) for v in names_class.values())

def test_names_class_negative():
    invalid_names_class = {0: 'person', '1': 42}
    with pytest.raises(AssertionError):
        assert isinstance(invalid_names_class, dict)
        assert all(isinstance(k, int) for k in invalid_names_class.keys())
        assert all(isinstance(v, str) for v in invalid_names_class.values())

@patch('object_detection.YOLO')
def test_draw_boxes_positive(MockYOLO):
    mock_model = MagicMock()
    MockYOLO.return_value = mock_model
    mock_result = [
        MagicMock(boxes=MagicMock(data=np.array([[0, 0, 10, 10, 0.9, 2]])))
    ]
    mock_model.return_value = mock_result

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    selected_classes = [2, 3]
    detections = detect_objects(image, mock_model, selected_classes)

    frame_with_boxes = draw_boxes(image, detections)
    assert frame_with_boxes.shape == image.shape

def test_draw_boxes_negative():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = []
    frame_with_boxes = draw_boxes(image, detections)
    assert frame_with_boxes.shape == image.shape
