import streamlit as st
import matplotlib.pyplot as plt
from itertools import permutations

PALLET_LENGTH = 120
PALLET_WIDTH = 80

def fit_boxes_on_pallet(box_length, box_width, box_height, box_count, pallet_max_height):
    orientations = set(permutations((box_length, box_width, box_height)))
    best_fit = {
        "boxes_per_layer": 0,
        "layers": 0,
        "total_fit": 0,
        "orientation": ()
    }

    for orientation in orientations:
        l, w, h = orientation
        if h > pallet_max_height:
            continue

        if l == 0 or w == 0 or h == 0:
            continue

        boxes_in_length = PALLET_LENGTH // l
        boxes_in_width = PALLET_WIDTH // w
        boxes_per_layer = boxes_in_length * boxes_in_width
        layers = pallet_max_height // h
        total_fit = boxes_per_layer * layers

        if total_fit > best_fit["total_fit"]:
            best_fit.update({
                "boxes_per_layer": boxes_per_layer,
                "layers": layers,
                "total_fit": total_fit,
                "orientation": orientation
            })

    if best_fit["total_fit"] == 0:
        return {
            "fit_count": 0,
            "leftover": box_count,
            "boxes_per_layer": 0,
            "layers": 0,
            "orientation": (0, 0, 0)
        }

    fit_count = min(box_count, best_fit["total_fit"])
    leftover = box_count - fit_count

    return {
        "fit_count": fit_count,
        "leftover": leftover,
        "boxes_per_layer": best_fit["boxes_per_layer"],
        "layers": best_fit["layers"],
        "orientation": best_fit["orientation"]
    }

def draw_pallet_layout(orientation, boxes_per_layer, layers):
    l, w, h = orientation
    if l == 0 or w == 0:
        st.write("Нет подходящей ориентации для отображения")
        return

    boxes_in_length = PALLET_LENGTH // l
    boxes_in_width = PALLET_WIDTH // w

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, PALLET_LENGTH)
    ax.set_ylim(0, PALLET_WIDTH)
    ax.set_title("Один слой коробов на паллете")

    for i in range(boxes_in_length):
        for j in range(boxes_in_width):
            rect = plt.Rectangle((i * l, j * w), l, w, linewidth=1, edgecolor='blue', facecolor='skyblue')
            ax.add_patch(rect)

    ax.set_aspect('equal')
    st.pyplot(fig)

st.title("Расчёт размещения коробов на паллете")

box_length = st.number_input("Длина короба (см)", min_value=1)
box_width = st.number_input("Ширина короба (см)", min_value=1)
box_height = st.number_input("Высота короба (см)", min_value=1)
box_count = st.number_input("Количество коробов", min_value=1)
pallet_max_height = st.number_input("Макс. высота паллеты (см)", min_value=1)

if st.button("Рассчитать"):
    result = fit_boxes_on_pallet(box_length, box_width, box_height, box_count, pallet_max_height)

    st.subheader("Результаты")
    st.write(f"Коробов поместилось: {result['fit_count']}")
    st.write(f"Осталось вне паллеты: {result['leftover']}")
    st.write(f"Количество слоёв: {result['layers']}")
    st.write(f"Коробов на одном слое: {result['boxes_per_layer']}")
    st.write(f"Использованная ориентация: {result['orientation']}")

    draw_pallet_layout(result['orientation'], result['boxes_per_layer'], result['layers'])
