import re
from pathlib import Path


# Define directory path and GitHub image URL prefix
github_url_prefix = 'https://raw.githubusercontent.com/JamboChen/Jambo-blog/master/'

# Compile regular expression pattern
img_pattern = re.compile(r'!\[(?P<alt_text>.*?)\]\((?P<image_path>.*?)\)')


def get_absolute_image_path(file_path, image_path):
    """Return the absolute path to the image file"""
    return str(file_path.parent.joinpath(image_path).resolve().relative_to(Path.cwd()))


def replace():
    for file_path in Path('./').glob('**/*.md'):
        with open(file_path, 'r') as f:
            content = f.read()

        for alt_text, image_path in img_pattern.findall(content):
            if not image_path.startswith('http'):
                img_url = github_url_prefix + str(file_path.parent.joinpath(image_path).resolve().relative_to(Path.cwd()))
                content = content.replace(f'![{alt_text}]({image_path})', f'![{alt_text}]({img_url})')

        with open(file_path, 'w') as f:
            f.write(content)


replace()
