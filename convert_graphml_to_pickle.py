#!/usr/bin/env python
import os
import pickle
import networkx as nx

def convert_graphml_to_pickle(input_path, output_path):
    """
    Загружает граф из файла GraphML и сохраняет его в формате pickle.
    """
    try:
        # Загружаем граф из GraphML
        G = nx.read_graphml(input_path)
        # Сохраняем граф в формате pickle
        with open(output_path, 'wb') as f:
            pickle.dump(G, f)
        print(f"Converted: {input_path} -> {output_path}")
    except Exception as e:
        print(f"Error converting {input_path}: {e}")

def process_directory(base_dir):
    """
    Рекурсивно обходит базовую директорию и для каждого файла с расширением
    .graphml создаёт рядом pickle-файл с сохранённой структурой графа.
    """
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".graphml"):
                input_file = os.path.join(root, file)
                # Заменяем расширение .graphml на .pkl
                output_file = os.path.join(root, os.path.splitext(file)[0] + ".pkl")
                convert_graphml_to_pickle(input_file, output_file)
    print("Batch processing complete.")

def main():
    # Укажите путь к базовой директории (замените на вашу директорию)
    base_dir = "E:/0_no_firrtl_bench/1 (100-139in, 100-134 out)"
    process_directory(base_dir)

if __name__ == '__main__':
    main()
