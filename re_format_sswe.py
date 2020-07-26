import config


old_sswe = config.WORD_EMBEDDING_DIR / 'sswe-u.txt'
new_sswe = config.WORD_EMBEDDING_DIR / 'new_sswe-u.txt'
if not new_sswe.exists():
    lines = []
    with old_sswe.open('r') as old_file:
        with new_sswe.open('w+') as new_file:
            for line in old_file:
                line_parts = line.split()
                new_line = ' '.join(line_parts)
                new_file.write(f'{new_line}\n')