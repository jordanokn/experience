import click


def remove_comment(line: str, all_comments: bool = True) -> str:
    """ пока не работает разделение на all_comments """
    comment_index = line.find('#')
    if comment_index != -1:
        if not any(quote in line[:comment_index] for quote in ('"', "'")):
            stripped_line = line[:comment_index].rstrip()
            if stripped_line:
                return stripped_line + '\n'
            return stripped_line
    return line


@click.command()
@click.option("--all_comments", default=True, help="что хочешь???")
@click.option("--file", help="мне вот откуда удалять коментарии а, щас все снесу?")
def clean_file_comments(
    file: str,
    all_comments: bool = True
):
    with open(file, "r", encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = [remove_comment(line, all_comments) for line in lines]
    
    with open(file, "w", encoding='utf-8') as f:
        f.writelines(cleaned_lines)


if __name__ == "__main__":
    clean_file_comments()
    
