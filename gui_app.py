import flet as ft
from model import ModulationSelector
import numpy as np

class ModulationSelectorGUI:
    def __init__(self):
        self.selector = ModulationSelector(model_type='mlp')
        try:
            self.selector.load_model('modulation_selector_model.h5')
            print("Модель успешно загружена")
        except:
            print("Не удалось загрузить модель. Используется ненатренированная модель.")

    def build(self, page: ft.Page):
        page.title = "Выбор модуляции"
        page.theme_mode = ft.ThemeMode.DARK
        page.padding = 20

        # Поля ввода
        self.snr_input = ft.TextField(
            label="Отношение сигнал/шум (дБ)",
            value="12.0",
            width=200
        )
        self.bandwidth_input = ft.TextField(
            label="Полоса пропускания (МГц)",
            value="3.0",
            width=200
        )
        self.data_rate_input = ft.TextField(
            label="Требуемая скорость передачи (Мбит/с)",
            value="6.0",
            width=200
        )
        self.interference_input = ft.TextField(
            label="Уровень помех (0-1)",
            value="0.4",
            width=200
        )
        self.power_input = ft.TextField(
            label="Ограничения по мощности (0-1)",
            value="0.6",
            width=200
        )
        self.ber_input = ft.Dropdown(
            label="Требования к BER",
            options=[
                ft.dropdown.Option("1e-3"),
                ft.dropdown.Option("1e-4"),
                ft.dropdown.Option("1e-5")
            ],
            value="1e-4",
            width=200
        )

        # Текст результатов
        self.results_text = ft.Text(size=16)

        def calculate_results(e):
            try:
                # Преобразование входных данных в нужные типы
                params = {
                    'snr': float(self.snr_input.value),
                    'bandwidth': float(self.bandwidth_input.value) * 1e6,  # Конвертация МГц в Гц
                    'required_data_rate': float(self.data_rate_input.value) * 1e6,  # Конвертация Мбит/с в бит/с
                    'interference_level': float(self.interference_input.value),
                    'power_constraints': float(self.power_input.value),
                    'ber_requirements': float(self.ber_input.value)
                }

                # Получение результатов от модели
                result = self.selector.select_modulation(params)

                # Форматирование результатов
                results_str = (
                    f"Выбранная модуляция: {result['selected_modulation']}\n"
                    f"Уверенность: {result['confidence']:.2%}\n\n"
                    "Все вероятности:\n"
                )
                for mod, prob in result['all_probabilities'].items():
                    results_str += f"  {mod}: {prob:.2%}\n"

                self.results_text.value = results_str
                page.update()

            except ValueError as e:
                self.results_text.value = f"Ошибка: Проверьте правильность введенных значений. {str(e)}"
                page.update()

        # Создание кнопки расчета
        calculate_button = ft.ElevatedButton(
            "Рассчитать модуляцию",
            on_click=calculate_results
        )

        # Компоновка
        page.add(
            ft.Column([
                ft.Text("Параметры выбора модуляции", size=20, weight=ft.FontWeight.BOLD),
                ft.Row([self.snr_input, self.bandwidth_input], spacing=20),
                ft.Row([self.data_rate_input, self.interference_input], spacing=20),
                ft.Row([self.power_input, self.ber_input], spacing=20),
                calculate_button,
                ft.Divider(),
                ft.Text("Результаты:", size=16, weight=ft.FontWeight.BOLD),
                self.results_text
            ], spacing=20)
        )

def main(page: ft.Page):
    app = ModulationSelectorGUI()
    app.build(page)

if __name__ == "__main__":
    ft.app(target=main) 