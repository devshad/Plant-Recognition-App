import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
import numpy as np
import json
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QTextEdit, QScrollArea, QLineEdit, QMessageBox, QTabWidget
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt
from tensorflow.keras.preprocessing import image
from PIL import Image
import sys
import pyttsx3
import re

# Image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Initialize pyttsx3 engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1)

# Build model
def build_model(num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load plant info
def load_plant_info(file_path='plants_info.json'):
    try:
        with open(file_path, 'r') as file:
            plant_info = json.load(file)
        print("Loaded plant_info:", plant_info)
        return plant_info
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {file_path}. {e}")
        return {}

# Check if image is leaf
def is_leaf_image(image):
    img_array = np.array(image)
    green_pixels = np.mean(img_array[:, :, 1])
    red_pixels = np.mean(img_array[:, :, 0])
    blue_pixels = np.mean(img_array[:, :, 2])
    return green_pixels > max(red_pixels, blue_pixels) and green_pixels > 50

# Preprocess image
def preprocess_image(image_path):
    image_obj = Image.open(image_path).resize(IMG_SIZE)
    img_array = np.array(image_obj) / 255.0
    return np.expand_dims(img_array, axis=0)

# Load dataset with enhanced data augmentation
def load_dataset(dataset_dir='dataset', img_size=(224, 224), batch_size=16):
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory {dataset_dir} not found.")
        return None, None, []
    
    plant_classes = sorted(os.listdir(dataset_dir))
    plant_counts = {}
    total_images = 0
    
    print(f"Found {len(plant_classes)} plant classes: {plant_classes}")
    for plant in plant_classes:
        plant_path = os.path.join(dataset_dir, plant)
        if os.path.isdir(plant_path):
            images = [f for f in os.listdir(plant_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            plant_counts[plant] = len(images)
            total_images += len(images)
            print(f"{plant}: {len(images)} images")
    
    print(f"Total: {total_images} images")
    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.4,
        channel_shift_range=20.0,  # Simulate lighting variations
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, validation_generator, plant_classes

# Train model with learning rate scheduler and more epochs
def train_model(model, train_generator, validation_generator):
    if train_generator is None or validation_generator is None:
        print("Cannot train: Dataset not loaded.")
        return model
    
    # Learning rate scheduler
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        callbacks=[lr_scheduler],
        verbose=1
    )
    model.save('plant_model.h5')
    print("Model saved as plant_model.h5")
    return model

# Load accuracy stats
def load_accuracy_stats(file_path='accuracy_stats.json'):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        # Initialize with zeros for all plant classes
        return {}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {file_path}. {e}")
        return {}

# Save accuracy stats
def save_accuracy_stats(stats, file_path='accuracy_stats.json'):
    with open(file_path, 'w') as file:
        json.dump(stats, file, indent=4)

# Update accuracy stats after a prediction
def update_accuracy_stats(predicted_class, is_correct, stats):
    if predicted_class not in stats:
        stats[predicted_class] = {"correct": 0, "total": 0}
    stats[predicted_class]["total"] += 1
    if is_correct:
        stats[predicted_class]["correct"] += 1
    save_accuracy_stats(stats)

# Save mistakes
def save_mistake(image_path, correct_label):
    if not os.path.exists('corrections'):
        os.makedirs('corrections')
    with open('corrections/mistakes.json', 'a') as f:
        f.write(json.dumps({'image_path': image_path, 'correct_label': correct_label}) + '\n')
    print(f"Mistake saved: {image_path} -> {correct_label}")

# Load correction history
def load_correction_history(file_path='corrections/mistakes.json'):
    corrections = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                corrections.append(json.loads(line.strip()))
        return corrections
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {file_path}. {e}")
        return []

# UI
class PlantRecognitionApp(QMainWindow):
    def __init__(self, model, plant_classes, plant_info):
        super().__init__()
        self.model = model
        self.plant_classes = plant_classes
        self.plant_info = plant_info
        self.current_image_path = None  # To track current file
        self.is_dark_mode = False  # Track theme state
        self.accuracy_stats = load_accuracy_stats()  # Load accuracy stats
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Plant Leaf Recognition")
        self.setGeometry(100, 100, 900, 750)

        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Define light and dark mode styles
        self.light_mode_styles = {
            "main_widget": """ 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #d1e2d0, stop:1 #a1b0a0);
            """,
            "title_label": """
                color: #E0E0E0;
                padding: 15px;
                background: #033b00;
                border-radius: 10px;
                margin: 10px;
            """,
            "info_text": """
                color: #E0E0E0;
                background: #8b917c;
                border: none;
                padding: 15px;
                border-radius: 10px;
                margin: 0 20px;
            """,
            "upload_btn": """
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    padding: 12px;
                    border-radius: 25px;
                    margin: 10px 150px;
                }
                QPushButton:hover {
                    background-color: #1e88e5;
                    transform: scale(1.05);
                }
                QPushButton:pressed {
                    background-color: #1e88e5;
                }
            """,
            "retrain_btn": """
                QPushButton {
                    background-color: #FF9800;
                    color: white;
                    padding: 12px;
                    border-radius: 25px;
                    margin: 10px 150px;
                }
                QPushButton:hover {
                    background-color: #e68900;
                    transform: scale(1.05);
                }
                QPushButton:pressed {
                    background-color: #cc7a00;
                }
            """,
            "results_area": """
                background: #e1eef5;
                color: #333333;
                padding: 20px;
                border-radius: 15px;
                margin: 20px 20px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            """,
            "stats_area": """
                background: #e1eef5;
                color: #333333;
                padding: 20px;
                border-radius: 15px;
                margin: 20px 20px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            """,
            "corrections_area": """
                background: #e1eef5;
                color: #333333;
                padding: 20px;
                border-radius: 15px;
                margin: 20px 20px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            """,
            "scroll_area": """
                QScrollArea {
                    border: none;
                    background: transparent;
                    margin: 0 20px;
                }
                QScrollBar:vertical {
                    border: none;
                    background: #f0f0f0;
                    width: 10px;
                    margin: 0;
                    border-radius: 5px;
                }
                QScrollBar::handle:vertical {
                    background: #4CAF50;
                    border-radius: 5px;
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                    height: 0;
                }
            """,
            "correction_input": """
                background-color: #fff;
                border: 2px solid #4CAF50;
                border-radius: 20px;
                padding: 10px;
                margin: 10px 20px;
            """,
            "correct_btn": """
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    padding: 12px;
                    border-radius: 25px;
                    margin: 10px 150px;
                }
                QPushButton:hover {
                    background-color: #1e88e5;
                    transform: scale(1.05);
                }
                QPushButton:pressed {
                    background-color: #1565c0;
                }
            """
        }

        self.dark_mode_styles = {
            "main_widget": """ 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #2e2e2e, stop:1 #1a1a1a);
            """,
            "title_label": """
                color: #E0E0E0;
                padding: 15px;
                background: #1a3c34;
                border-radius: 10px;
                margin: 10px;
            """,
            "info_text": """
                color: #E0E0E0;
                background: #3b3b3b;
                border: none;
                padding: 15px;
                border-radius: 10px;
                margin: 0 20px;
            """,
            "upload_btn": """
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    padding: 12px;
                    border-radius: 25px;
                    margin: 10px 150px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                    transform: scale(1.05);
                }
                QPushButton:pressed {
                    background-color: #3d8b40;
                }
            """,
            "retrain_btn": """
                QPushButton {
                    background-color: #FF9800;
                    color: white;
                    padding: 12px;
                    border-radius: 25px;
                    margin: 10px 150px;
                }
                QPushButton:hover {
                    background-color: #e68900;
                    transform: scale(1.05);
                }
                QPushButton:pressed {
                    background-color: #cc7a00;
                }
            """,
            "results_area": """
                background: #2e2e2e;
                color: #E0E0E0;
                padding: 20px;
                border-radius: 15px;
                margin: 20px 20px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
            """,
            "stats_area": """
                background: #2e2e2e;
                color: #E0E0E0;
                padding: 20px;
                border-radius: 15px;
                margin: 20px 20px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
            """,
            "corrections_area": """
                background: #2e2e2e;
                color: #E0E0E0;
                padding: 20px;
                border-radius: 15px;
                margin: 20px 20px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
            """,
            "scroll_area": """
                QScrollArea {
                    border: none;
                    background: transparent;
                    margin: 0 20px;
                }
                QScrollBar:vertical {
                    border: none;
                    background: #444444;
                    width: 10px;
                    margin: 0;
                    border-radius: 5px;
                }
                QScrollBar::handle:vertical {
                    background: #4CAF50;
                    border-radius: 5px;
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                    height: 0;
                }
            """,
            "correction_input": """
                background-color: #444444;
                color: #E0E0E0;
                border: 2px solid #4CAF50;
                border-radius: 20px;
                padding: 10px;
                margin: 10px 20px;
            """,
            "correct_btn": """
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    padding: 12px;
                    border-radius: 25px;
                    margin: 10px 150px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                    transform: scale(1.05);
                }
                QPushButton:pressed {
                    background-color: #3d8b40;
                }
            """
        }

        # Apply light mode by default
        main_widget.setStyleSheet(self.light_mode_styles["main_widget"])

        title_label = QLabel("🌿 Plant Leaf Recognition", self)
        title_label.setFont(QFont('Roboto', 28, QFont.Bold))
        title_label.setStyleSheet(self.light_mode_styles["title_label"])
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        info_text = QTextEdit(self)
        info_text.setReadOnly(True)
        info_text.setText(
            "Discover the wonders of nature with AI!\n"
            "Brought to you by Team Nebula\n"
            "Department of Information and Communication Engineering,"
            " Daffodil International University.\n"
            "Snap or upload a leaf — let AI uncover the secrets of the plant world!"
        )
        info_text.setFont(QFont('Roboto', 12))
        info_text.setStyleSheet(self.light_mode_styles["info_text"])
        info_text.setFixedHeight(110)
        layout.addWidget(info_text)

        # Dark mode toggle button
        self.theme_btn = QPushButton("Switch to Dark Mode", self)
        self.theme_btn.setFont(QFont('Roboto', 12, QFont.Bold))
        self.theme_btn.setStyleSheet("""
            QPushButton {
                background-color: #666;
                color: white;
                padding: 8px;
                border-radius: 15px;
                margin: 5px 300px;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QPushButton:pressed {
                background-color: #444;
            }
        """)
        self.theme_btn.clicked.connect(self.toggle_theme)
        layout.addWidget(self.theme_btn)

        upload_btn = QPushButton("Upload Leaf Images", self)
        upload_btn.setFont(QFont('Roboto', 14, QFont.Bold))
        upload_btn.setStyleSheet(self.light_mode_styles["upload_btn"])
        upload_btn.clicked.connect(self.upload_images)
        layout.addWidget(upload_btn)

        # Retrain model button
        retrain_btn = QPushButton("Retrain Model", self)
        retrain_btn.setFont(QFont('Roboto', 14, QFont.Bold))
        retrain_btn.setStyleSheet(self.light_mode_styles["retrain_btn"])
        retrain_btn.clicked.connect(self.retrain_model)
        layout.addWidget(retrain_btn)

        # Tab widget for results, stats, and corrections
        self.tab_widget = QTabWidget(self)
        self.tab_widget.setFont(QFont('Roboto', 12))

        # Results tab
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        self.results_area = QTextEdit(self)
        self.results_area.setReadOnly(True)
        self.results_area.setFont(QFont('Consolas', 11))
        self.results_area.setStyleSheet(self.light_mode_styles["results_area"])
        scroll = QScrollArea()
        scroll.setWidget(self.results_area)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(self.light_mode_styles["scroll_area"])
        results_layout.addWidget(scroll)
        self.tab_widget.addTab(results_widget, "Predictions")

        # Stats tab
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        self.stats_area = QTextEdit(self)
        self.stats_area.setReadOnly(True)
        self.stats_area.setFont(QFont('Consolas', 11))
        self.stats_area.setStyleSheet(self.light_mode_styles["stats_area"])
        stats_scroll = QScrollArea()
        stats_scroll.setWidget(self.stats_area)
        stats_scroll.setWidgetResizable(True)
        stats_scroll.setStyleSheet(self.light_mode_styles["scroll_area"])
        stats_layout.addWidget(stats_scroll)
        self.tab_widget.addTab(stats_widget, "Accuracy Stats")

        # Corrections tab
        corrections_widget = QWidget()
        corrections_layout = QVBoxLayout(corrections_widget)
        self.corrections_area = QTextEdit(self)
        self.corrections_area.setReadOnly(True)
        self.corrections_area.setFont(QFont('Consolas', 11))
        self.corrections_area.setStyleSheet(self.light_mode_styles["corrections_area"])
        corrections_scroll = QScrollArea()
        corrections_scroll.setWidget(self.corrections_area)
        corrections_scroll.setWidgetResizable(True)
        corrections_scroll.setStyleSheet(self.light_mode_styles["scroll_area"])
        corrections_layout.addWidget(corrections_scroll)
        self.tab_widget.addTab(corrections_widget, "Correction History")

        layout.addWidget(self.tab_widget)

        # Correction input
        self.correction_input = QLineEdit(self)
        self.correction_input.setPlaceholderText("✍️ Not the right plant? Help us out — correct the plant name! ")
        self.correction_input.setFont(QFont('Roboto', 12))
        self.correction_input.setStyleSheet(self.light_mode_styles["correction_input"])
        layout.addWidget(self.correction_input)

        correct_btn = QPushButton("Submit Correction", self)
        correct_btn.setFont(QFont('Roboto', 14, QFont.Bold))
        correct_btn.setStyleSheet(self.light_mode_styles["correct_btn"])
        correct_btn.clicked.connect(self.submit_correction)
        layout.addWidget(correct_btn)

        layout.addStretch()

        # Store references to widgets for theme toggling
        self.main_widget = main_widget
        self.title_label = title_label
        self.info_text = info_text
        self.upload_btn = upload_btn
        self.retrain_btn = retrain_btn
        self.scroll = scroll
        self.stats_scroll = stats_scroll
        self.corrections_scroll = corrections_scroll
        self.correct_btn = correct_btn

        # Initialize stats and corrections display
        self.update_stats_display()
        self.update_corrections_display()

    def toggle_theme(self):
        self.is_dark_mode = not self.is_dark_mode
        styles = self.dark_mode_styles if self.is_dark_mode else self.light_mode_styles

        # Update button text
        self.theme_btn.setText("Switch to Light Mode" if self.is_dark_mode else "Switch to Dark Mode")

        # Apply styles to all widgets
        self.main_widget.setStyleSheet(styles["main_widget"])
        self.title_label.setStyleSheet(styles["title_label"])
        self.info_text.setStyleSheet(styles["info_text"])
        self.upload_btn.setStyleSheet(styles["upload_btn"])
        self.retrain_btn.setStyleSheet(styles["retrain_btn"])
        self.results_area.setStyleSheet(styles["results_area"])
        self.stats_area.setStyleSheet(styles["stats_area"])
        self.corrections_area.setStyleSheet(styles["corrections_area"])
        self.scroll.setStyleSheet(styles["scroll_area"])
        self.stats_scroll.setStyleSheet(styles["scroll_area"])
        self.corrections_scroll.setStyleSheet(styles["scroll_area"])
        self.correction_input.setStyleSheet(styles["correction_input"])
        self.correct_btn.setStyleSheet(styles["correct_btn"])

        # Adjust results area text color dynamically
        current_html = self.results_area.toHtml()
        if self.is_dark_mode:
            current_html = current_html.replace('color:#333333;', 'color:#E0E0E0;')
            current_html = current_html.replace('background:#F0FFF0;', 'background:#3e3e3e;')
        else:
            current_html = current_html.replace('color:#E0E0E0;', 'color:#333333;')
            current_html = current_html.replace('background:#3e3e3e;', 'background:#F0FFF0;')
        self.results_area.setHtml(current_html)

        # Adjust stats and corrections area text color
        self.update_stats_display()
        self.update_corrections_display()

    def update_stats_display(self):
        stats_text = "<h2>Accuracy Statistics</h2>"
        for plant, stats in self.accuracy_stats.items():
            total = stats["total"]
            correct = stats["correct"]
            accuracy = (correct / total * 100) if total > 0 else 0
            plant_display = plant.replace('_', ' ')
            stats_text += f"<p><b>{plant_display}</b>: {correct}/{total} correct ({accuracy:.2f}%)</p>"
        self.stats_area.setHtml(stats_text)

    def update_corrections_display(self):
        corrections = load_correction_history()
        corrections_text = "<h2>Correction History</h2>"
        if not corrections:
            corrections_text += "<p>No corrections recorded yet.</p>"
        else:
            for correction in corrections:
                image_path = correction['image_path']
                correct_label = correction['correct_label']
                corrections_text += f"<p><b>Image:</b> {os.path.basename(image_path)}<br><b>Corrected to:</b> {correct_label}</p>"
        self.corrections_area.setHtml(corrections_text)

    def retrain_model(self):
        QMessageBox.information(self, "Retraining", "Starting model retraining... This may take a while.")
        train_generator, validation_generator, _ = load_dataset()
        if train_generator and validation_generator:
            self.model = build_model(len(self.plant_classes))
            self.model = train_model(self.model, train_generator, validation_generator)
            QMessageBox.information(self, "Success", "Model retraining completed and saved as plant_model.h5!")
        else:
            QMessageBox.warning(self, "Error", "Failed to load dataset for retraining.")

    def upload_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Leaf Images", "", "Images (*.jpg *.png)")
        if not files:
            self.results_area.setText("No images selected.")
            return

        results = ""
        for file in files:
            image_obj = Image.open(file)
            if not is_leaf_image(image_obj):
                results += f"<span style='color:red;'>❌ {os.path.basename(file)}: Not a leaf image!</span><br><br>"
                continue
            self.current_image_path = file
            prediction_result = self.predict_image(file)
            results += prediction_result

            # Update accuracy stats (assume correct until user corrects)
            predicted_class = self.last_prediction
            update_accuracy_stats(predicted_class, True, self.accuracy_stats)
            self.update_stats_display()

        self.results_area.setHtml(results)
        self.speak_result(results)

    def predict_image(self, image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = self.model.predict(img_array)
        self.last_prediction = self.plant_classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Replace underscores with spaces for display
        plant_name_display = self.last_prediction.replace('_', ' ')

        info = self.plant_info.get(self.last_prediction, {})
        scientific_name = info.get('scientific_name', 'Unknown')
        origin = info.get('origin_and_discovery', 'Unknown')
        bengali_name = info.get('bengali_name', 'Unknown')
        global_distribution = info.get('global_distribution', 'Unknown')
        usage = info.get('usage', ['No usage available.'])
        care_tips = info.get('care_tips', ['No care tips available.'])

        result = f"""
        <div style="background:#F0FFF0;padding:15px;margin-bottom:15px;border-radius:10px;">
            <h2 style="color:#2E8B57;">🌿 {plant_name_display}</h2>
            <p><b>Confidence:</b> {confidence:.2f}%</p>
            <p><b>Scientific Name:</b> {scientific_name}</p>
            <p><b>Origin and Discovery:</b> {origin}</p>
            <p><b>Bengali Name:</b> {bengali_name}</p>
            <p><b>Global Distribution:</b> {global_distribution}</p>
            <p><b>Usage:</b><br>""" + "<br>".join(f"🔹 {use}" for use in usage) + """</p>
            <p><b>Care Tips:</b><br>""" + "<br>".join(f"💧 {tip}" for tip in care_tips) + """</p>
        </div>
        """
        return result

    def speak_result(self, result):
        # Strip HTML tags for text-to-speech
        clean_text = re.sub(r'<[^>]+>', '', result)  # Remove HTML tags
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()  # Normalize whitespace
        clean_text = clean_text.replace('🔹', '-')  # Replace usage emoji with dash
        clean_text = clean_text.replace('💧', '-')  # Replace care tips emoji with dash
        engine.say(clean_text)
        engine.runAndWait()

    def submit_correction(self):
        correct_text = self.correction_input.text().strip()
        if correct_text and self.current_image_path:
            save_mistake(self.current_image_path, correct_text)
            
            # Update accuracy stats: mark the previous prediction as incorrect
            if self.last_prediction:
                predicted_class = self.last_prediction
                # Decrement the correct count for the predicted class
                if predicted_class in self.accuracy_stats:
                    self.accuracy_stats[predicted_class]["correct"] = max(0, self.accuracy_stats[predicted_class]["correct"] - 1)
                self.update_stats_display()

            QMessageBox.information(self, "Correction Saved", "Thanks! Your correction has been recorded.")
            self.correction_input.clear()
            self.update_corrections_display()
        else:
            QMessageBox.warning(self, "Error", "Please upload an image and enter the correct plant name!")

# Main function
def main():
    # Dynamically load plant classes from dataset directory
    dataset_dir = 'dataset'
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory {dataset_dir} not found. Exiting.")
        return
    
    plant_classes = sorted(os.listdir(dataset_dir))
    print(f"Found {len(plant_classes)} plant classes: {plant_classes}")

    # Load dataset and train
    #train_generator, validation_generator, _ = load_dataset()
    #if plant_classes:
    #    model = build_model(len(plant_classes))
    #    model = train_model(model, train_generator, validation_generator)
    #else:
    #    print("No plant classes found. Exiting.")
    #    return

    # Load saved model and plant info
    model = tf.keras.models.load_model('plant_model.h5')
    plant_info = load_plant_info()

    app = QApplication(sys.argv)
    window = PlantRecognitionApp(model, plant_classes, plant_info)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()