import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class ModulationSelector:
    def __init__(self, model_type='mlp', input_shape=6, num_classes=5):
        """ Инициализация селектора модуляции
        Args:
            model_type (str): 'mlp' или 'cnn'
            input_shape (int): количество входных параметров
            num_classes (int): количество методов модуляции
        """
        self.model_type = model_type
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.modulations = ['BPSK', 'QPSK', '8-PSK', '16-QAM', '64-QAM']
        
        if model_type == 'mlp':
            self.model = self._create_mlp_model()
        elif model_type == 'cnn':
            self.model = self._create_cnn_model()
        else:
            raise ValueError("Supported model types: 'mlp' or 'cnn'")
    
    def _create_mlp_model(self):
        """ Создание MLP модели """
        model = Sequential([
            Dense(128, activation='relu', input_shape=(self.input_shape,)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
    
    def _create_cnn_model(self):
        """ Создание CNN модели (для временных рядов/спектров) """
        model = Sequential([
            Reshape((self.input_shape, 1), input_shape=(self.input_shape,)),
            
            Conv1D(32, 3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(2),
            
            Conv1D(64, 3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(2),
            
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def generate_data(self, num_samples=10000):
        """ Генерация синтетических данных для обучения """
        X = np.zeros((num_samples, self.input_shape))
        y = np.zeros((num_samples, self.num_classes))
        
        for i in range(num_samples):
            snr = np.random.uniform(0, 30)
            bandwidth = np.random.uniform(1e6, 10e6)
            required_rate = np.random.uniform(1e6, 20e6)
            interference = np.random.uniform(0, 1)
            power = np.random.uniform(0, 1)
            ber = np.random.choice([1e-3, 1e-4, 1e-5])
            
            X[i] = [
                snr / 30.0,                  # Нормализованный SNR (0-30 dB)
                bandwidth / 10e6,            # Нормализованная полоса (1-10 MHz)
                required_rate / 20e6,         # Нормализованная скорость (1-20 Mbps)
                interference,                # Уровень помех (0-1)
                power,                       # Ограничения мощности (0-1)
                ber * 1e4                    # Нормализованные требования BER
            ]
            
            # Простые правила для генерации меток
            if snr < 5:
                y[i, 0] = 1  # BPSK
            elif snr < 10:
                y[i, 1] = 1  # QPSK
            elif snr < 15:
                y[i, 2] = 1  # 8-PSK
            elif snr < 20:
                y[i, 3] = 1  # 16-QAM
            else:
                y[i, 4] = 1  # 64-QAM
        
        return X, y
    
    def train(self, X=None, y=None, epochs=50, batch_size=32, test_size=0.2):
        """ Обучение модели """
        if X is None or y is None:
            X, y = self.generate_data()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Графики обучения
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.show()
        
        return history
    
    def select_modulation(self, input_params):
        """ Выбор метода модуляции на основе входных параметров
        Args:
            input_params (dict): Словарь с параметрами:
                {
                    'snr': float,             # Отношение сигнал/шум (dB)
                    'bandwidth': float,       # Полоса пропускания (Hz)
                    'required_data_rate': float,  # Требуемая скорость (bps)
                    'interference_level': float,   # Уровень помех (0-1)
                    'power_constraints': float,    # Ограничения мощности (0-1)
                    'ber_requirements': float      # Требования к BER (например 1e-4)
                }
        Returns:
            dict: Результаты выбора модуляции
        """
        # Нормализация входных данных
        input_array = np.array([
            input_params['snr'] / 30.0,
            input_params['bandwidth'] / 10e6,
            input_params['required_data_rate'] / 20e6,
            input_params['interference_level'],
            input_params['power_constraints'],
            input_params['ber_requirements'] * 1e4
        ]).reshape(1, -1)
        
        # Получение предсказания
        probabilities = self.model.predict(input_array, verbose=0)[0]
        modulation_index = np.argmax(probabilities)
        
        return {
            'selected_modulation': self.modulations[modulation_index],
            'confidence': float(probabilities[modulation_index]),
            'all_probabilities': {mod: float(prob) for mod, prob in zip(self.modulations, probabilities)}
        }
    
    def save_model(self, filepath):
        """ Сохранение модели в файл """
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """ Загрузка модели из файла """
        self.model = tf.keras.models.load_model(filepath)

# Пример использования
if __name__ == "__main__":
    # 1. Инициализация селектора
    selector = ModulationSelector(model_type='mlp')
    
    # 2. Обучение модели (можно пропустить, если загружаем сохраненную модель)
    print("Training the model...")
    selector.train(epochs=30)
    
    # 3. Пример выбора модуляции
    test_params = {
        'snr': 18.0,
        'bandwidth': 1e6,
        'required_data_rate': 8e6,
        'interference_level': 0.2,
        'power_constraints': 0.7,
        'ber_requirements': 1e-5
    }
    
    result = selector.select_modulation(test_params)
    print("\nModulation selection results:")
    print(f"Selected modulation: {result['selected_modulation']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("All probabilities:")
    for mod, prob in result['all_probabilities'].items():
        print(f"  {mod}: {prob:.2%}")
    
    # 4. Сохранение модели (опционально)
    selector.save_model('modulation_selector_model.h5')