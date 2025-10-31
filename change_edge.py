import re

# Путь к исходному файлу и файлу с результатами
input_file_path = "C:/Users/Arkadiy/Desktop/python_files/Myfuturejob/DIPLOM_AI/1 (100-139in, 100-134 out)/1 (100-139in, 100-134 out)/CCGRCG02026/CCGRCG02026.graphml"
output_file_path = "C:/Users/Arkadiy/Desktop/python_files/Myfuturejob/DIPLOM_AI/1 (100-139in, 100-134 out)/1 (100-139in, 100-134 out)/CCGRCG02026/CCGRCG02026_fixed.graphml"

# Открываем исходный файл и создаем новый файл с исправлениями
with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        # Ищем строки с <source=... и заменяем на <edge source=...>
        fixed_line = re.sub(r'<source="(.*?)" target="(.*?)"\s*/>', r'<edge source="\1" target="\2"/>', line)
        outfile.write(fixed_line)

print(f"Файл успешно исправлен и сохранён в {output_file_path}")
