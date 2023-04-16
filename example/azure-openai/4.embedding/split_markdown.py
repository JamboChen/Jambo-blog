import os
import pandas

MAX_LEN = 2048


def split_text(text, max_length=2048):
    paragraphs = text.split("\n")
    result = []
    current_paragraph = ""
    for paragraph in paragraphs:
        if len(current_paragraph) + len(paragraph) > max_length:
            result.append(current_paragraph)
            current_paragraph = paragraph
        else:
            current_paragraph += "\n" + paragraph
    if current_paragraph:
        result.append(current_paragraph)
    return result


def find_md_files(directory):
    result = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                result.append(os.path.join(root, file))
    return result


if __name__ == "__main__":
    df = pandas.DataFrame(columns=["file", "content"])
    for file in find_md_files("."):
        with open(file) as f:
            text = f.read()
        for c in split_text(text, MAX_LEN):
            df.loc[len(df)] = [file, c]

    df.to_csv("output.csv", index=False)
