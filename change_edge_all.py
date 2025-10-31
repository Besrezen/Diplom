import os
import re

# Путь к главной директории с подпапками
base_dir = "D:/0_no_firrtl_bench/5_2 (90_100 in, 50-100 out)"

# Функция для обработки файла
def fix_graphml_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile:
        content = infile.readlines()  # Считываем содержимое файла
    fixed_content = []
    for line in content:
        # Заменяем строки вида <source="..." target="..."/> на <edge source="..." target="..."/>
        fixed_line = re.sub(r'<source="(.*?)" target="(.*?)"\s*/>', r'<edge source="\1" target="\2"/>', line)
        fixed_content.append(fixed_line)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.writelines(fixed_content)  # Записываем исправленное содержимое

# Проход по всем подпапкам
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".graphml"):
            input_file = os.path.join(root, file)
            output_file = os.path.join(root, file)  # Сохраняем с тем же именем

            # Проверяем содержимое перед записью
            try:
                fix_graphml_file(input_file, output_file)
                print(f"Processed: {input_file}")
            except Exception as e:
                print(f"Error processing {input_file}: {e}")

print("Batch processing complete.")
