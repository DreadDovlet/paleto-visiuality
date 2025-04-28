import streamlit as st
import matplotlib.pyplot as plt
from itertools import permutations
import pandas as pd
import io
import os

# Константы
PALLET_LENGTH = 120
PALLET_WIDTH = 80
MAX_BOX_TYPES = 6

# Установка цвета фона с помощью CSS
st.markdown(
    """
    <style>
    body {
        background-color: #F0F0F0; /* Светло-серый фон, можно изменить */
    }
    </style>
    """,
    unsafe_allow_html=True
)

def fit_boxes_on_pallet(boxes, pallet_max_height, allow_full_rotation=True):
    """
    Рассчитывает, сколько коробов разных типов можно разместить на паллете.
    
    Args:
        boxes (list): Список словарей с данными коробов: [{'length': float, 'width': float, 'height': float, 'count': int}, ...]
        pallet_max_height (float): Максимальная высота паллеты в см.
        allow_full_rotation (bool): Разрешать ли полное вращение коробов.
    
    Returns:
        dict: Результаты: общее количество помещенных коробов, остаток, слои и ориентации.
    """
    # Валидация входных данных
    for box in boxes:
        if any(x <= 0 for x in [box['length'], box['width'], box['height'], box['count']]):
            return {
                "fit_count": 0,
                "leftover": sum(b['count'] for b in boxes),
                "layers": [],
                "total_boxes_per_layer": 0
            }
    
    if pallet_max_height <= 0:
        return {
            "fit_count": 0,
            "leftover": sum(b['count'] for b in boxes),
            "layers": [],
            "total_boxes_per_layer": 0
        }

    # Результаты для каждого типа коробов
    box_fits = []
    for box in boxes:
        orientations = set(permutations((box['length'], box['width'], box['height']))) if allow_full_rotation else \
                      {(box['length'], box['width'], box['height']), (box['width'], box['length'], box['height'])}
        
        best_fit = {
            "boxes_per_layer": 0,
            "boxes_in_length": 0,
            "boxes_in_width": 0,
            "layers": 0,
            "total_fit": 0,
            "orientation": (0, 0, 0),
            "box_index": boxes.index(box)
        }

        for orientation in orientations:
            l, w, h = orientation
            if h > pallet_max_height:
                continue

            boxes_in_length = PALLET_LENGTH // l
            boxes_in_width = PALLET_WIDTH // w
            boxes_per_layer = boxes_in_length * boxes_in_width
            layers = pallet_max_height // h
            total_fit = boxes_per_layer * layers

            if total_fit > best_fit["total_fit"]:
                best_fit.update({
                    "boxes_per_layer": boxes_per_layer,
                    "boxes_in_length": boxes_in_length,
                    "boxes_in_width": boxes_in_width,
                    "layers": layers,
                    "total_fit": total_fit,
                    "orientation": orientation
                })

        box_fits.append(best_fit)

    # Комбинируем слои для всех типов коробов
    total_height_used = 0
    layers = []
    total_fit_count = 0
    leftover = {i: boxes[i]['count'] for i in range(len(boxes))}

    # Пробуем размещать слои всех типов коробов, пока есть место и коробы
    while total_height_used < pallet_max_height and any(leftover[i] > 0 for i in range(len(boxes))):
        best_layer = None
        best_score = -1
        best_box_index = None

        # Проверяем каждый тип короба
        for fit in box_fits:
            box_index = fit['box_index']
            if leftover[box_index] <= 0 or fit['total_fit'] == 0:
                continue

            l, w, h = fit['orientation']
            if total_height_used + h > pallet_max_height:
                continue

            boxes_per_layer = fit['boxes_per_layer']
            boxes_in_length = fit['boxes_in_length']
            boxes_in_width = fit['boxes_in_width']

            # Оцениваем слой по количеству коробов
            score = min(leftover[box_index], boxes_per_layer)
            if score > best_score:
                best_score = score
                best_layer = {
                    "box_index": box_index,
                    "boxes_per_layer": boxes_per_layer,
                    "boxes_in_length": boxes_in_length,
                    "boxes_in_width": boxes_in_width,
                    "orientation": fit['orientation']
                }
                best_box_index = box_index

        if best_layer is None:
            break

        # Добавляем лучший слой
        layers.append(best_layer)
        total_height_used += best_layer['orientation'][2]  # Высота слоя
        total_fit_count += min(leftover[best_box_index], best_layer['boxes_per_layer'])
        leftover[best_box_index] -= min(leftover[best_box_index], best_layer['boxes_per_layer'])

    return {
        "fit_count": total_fit_count,
        "leftover": sum(leftover.values()),
        "layers": layers,
        "total_boxes_per_layer": sum(layer['boxes_per_layer'] for layer in layers)
    }

def draw_pallet_layout(layers):
    """
    Визуализирует 2D-раскладку первого слоя коробов на паллете.
    
    Args:
        layers (list): Список слоев с данными о коробах.
    
    Returns:
        BytesIO: Буфер с изображением PNG для скачивания.
    """
    if not layers:
        st.error("Невозможно отобразить раскладку: нет подходящих коробов.")
        return None

    # Берем первый слой для визуализации
    layer = layers[0]
    l, w, h = layer['orientation']
    boxes_per_layer = layer['boxes_per_layer']
    boxes_in_length = layer['boxes_in_length']
    boxes_in_width = layer['boxes_in_width']
    
    if boxes_per_layer > 1000:
        st.warning("Слишком много коробов для отображения. Визуализация отключена.")
        return None

    # Создаем 2D-график
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, PALLET_LENGTH)
    ax.set_ylim(0, PALLET_WIDTH)
    ax.set_title(f"Слой коробов типа {layer['box_index'] + 1} ({boxes_in_length}×{boxes_in_width})")

    # Рисуем коробы
    for i in range(boxes_in_length):
        for j in range(boxes_in_width):
            rect = plt.Rectangle((i * l, j * w), l, w, linewidth=1, edgecolor='blue', facecolor='skyblue')
            ax.add_patch(rect)

    ax.set_aspect('equal')

    # Отображаем график в Streamlit
    st.pyplot(fig)

    # Сохраняем изображение в буфер для скачивания
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)

    # Закрываем фигуру
    plt.close(fig)

    return buffer

# Streamlit интерфейс
# Добавление логотипа
logo_path = "logo.png"  # Укажите путь к вашему логотипу
if os.path.exists(logo_path):
    st.image(logo_path, width=200)  # Установите желаемую ширину логотипа
else:
    st.warning("Файл логотипа (logo.png) не найден. Убедитесь, что он находится в той же директории, что и скрипт.")

st.title("Расчёт размещения разных коробов на паллете")

# Пример использования
st.markdown("**Пример**: Короб 1: 40×30×20 см (5 шт.), Короб 2: 30×20×10 см (10 шт.), паллета 120×80×150 см.")

# Управление количеством типов коробов
if 'box_types' not in st.session_state:
    st.session_state.box_types = 1

def add_box_type():
    if st.session_state.box_types < MAX_BOX_TYPES:
        st.session_state.box_types += 1
    else:
        st.warning("Достигнуто максимальное количество типов коробов (6).")

def remove_box_type():
    if st.session_state.box_types > 1:
        st.session_state.box_types -= 1

st.button("Добавить тип короба", on_click=add_box_type)
st.button("Удалить тип короба", on_click=remove_box_type)

# Форма для ввода данных о коробах
boxes = []
for i in range(st.session_state.box_types):
    st.subheader(f"Короб типа {i + 1}")
    col1, col2 = st.columns(2)
    with col1:
        length = st.number_input(f"Длина короба {i + 1} (см)", min_value=1, step=1, value=40, key=f"length_{i}")
        width = st.number_input(f"Ширина короба {i + 1} (см)", min_value=1, step=1, value=30, key=f"width_{i}")
    with col2:
        height = st.number_input(f"Высота короба {i + 1} (см)", min_value=1, step=1, value=20, key=f"height_{i}")
        count = st.number_input(f"Количество коробов {i + 1}", min_value=1, step=1, value=5, key=f"count_{i}")
    boxes.append({"length": length, "width": width, "height": height, "count": count})

# Общие параметры
pallet_max_height = st.number_input("Макс. высота паллеты (см)", min_value=1, step=1, value=150)
allow_full_rotation = st.checkbox("Разрешить полное вращение коробов", value=True)

# Кнопки
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    calculate = st.button("Рассчитать")
with col_btn2:
    clear = st.button("Очистить")

if clear:
    st.session_state.box_types = 1
    st.rerun()

if calculate:
    result = fit_boxes_on_pallet(boxes, pallet_max_height, allow_full_rotation)

    # Вывод результатов
    st.subheader("Результаты")
    if result["fit_count"] == 0:
        st.error("Невозможно разместить короба с заданными параметрами.")
    else:
        st.write(f"**Всего коробов поместилось**: {result['fit_count']}")
        st.write(f"**Осталось вне паллеты**: {result['leftover']}")
        for layer in result['layers']:
            st.write(f"Слой коробов типа {layer['box_index'] + 1}: {layer['boxes_per_layer']} коробов "
                     f"({layer['boxes_in_length']} по длине × {layer['boxes_in_width']} по ширине), "
                     f"ориентация (Д×Ш×В): {layer['orientation']}")

        # Визуализация первого слоя и скачивание PNG
        image_buffer = draw_pallet_layout(result['layers'])
        if image_buffer:
            st.download_button(
                label="Скачать визуализацию (PNG)",
                data=image_buffer,
                file_name="pallet_layout.png",
                mime="image/png"
            )

        # Экспорт результатов в CSV
        result_data = [{
            "Тип короба": f"Тип {layer['box_index'] + 1}",
            "Коробов в слое": layer['boxes_per_layer'],
            "Коробов по длине": layer['boxes_in_length'],
            "Коробов по ширине": layer['boxes_in_width'],
            "Ориентация (Д×Ш×В)": layer['orientation']
        } for layer in result['layers']]
        result_data.append({
            "Тип короба": "Итого",
            "Коробов в слое": result['fit_count'],
            "Коробов по длине": "-",
            "Коробов по ширине": "-",
            "Ориентация (Д×Ш×В)": "-"
        })
        result_df = pd.DataFrame(result_data)
        csv_buffer = io.StringIO()
        result_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Скачать результаты в CSV",
            data=csv_buffer.getvalue(),
            file_name="pallet_results.csv",
            mime="text/csv"
        )
