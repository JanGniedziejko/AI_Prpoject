import os

# Mapa znaków do zastąpienia
replacements = {
    'ą':'a', 'æ':'c', 'ę':'e', 'ł':'l', 'ń':'n',
    'ó':'o', 'ś':'s', 'ż':'z', 'ź':'z',
    '³':'l', '¿':'z', 'ê':'e', '¹':'a', 'ú':'s', 'ô':'o', 'ñ':'n', 'ä':'a', 'ó':'o'
}

def remove_special_chars(text):
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text

num_of_authors = 8  # liczba autorów

for author_no in range(1, num_of_authors + 1):
    input_file = f"author{author_no}/word_places.txt"
    output_file = f"author{author_no}/word_places_clean.txt"

    if not os.path.exists(input_file):
        print(f"Plik nie istnieje: {input_file}")
        continue

    with open(input_file, "r", encoding="latin-1") as f:
        lines = f.readlines()

    cleaned_lines = [remove_special_chars(line) for line in lines]

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)

    print(f"Plik wyczyszczony i zapisany: {output_file}")
