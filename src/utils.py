from os import listdir, makedirs
from os.path import isfile, isdir, join, exists, dirname


def file_listing(dir, extension=None):
    files = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]
    if extension:
        files = list(filter(lambda f: f.endswith('.' + extension), files))
    return files


def dir_listing(base_dir):
    return [join(base_dir, d) for d in listdir(base_dir) if isdir(join(base_dir, d))]


def mkdir(path):
    if not exists(path):
        makedirs(path)


def last_component(path):
    return path.split('/')[-1]


def file_exists(path):
    return isfile(path)


def relative_path(path):
    base_dir = dirname(__file__)
    return join(base_dir, path)


def get_file_name(filepath):
    return last_component(filepath).split('.')[-2]
