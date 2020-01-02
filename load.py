import os

from lxml import objectify
import pandas as pd

# dev paths
dev_audio = './data/CoE_dataset/Dev_Set/audio_descriptors/'
dev_text = './data/CoE_dataset/Dev_Set/text_descriptors/'
dev_vis = './data/CoE_dataset/Dev_Set/vis_descriptors/'
dev_xml = './data/CoE_dataset/Dev_Set/XML/'

# test paths
test_audio = './data/CoE_dataset/Test_Set/audio_descriptors/'
test_text = './data/CoE_dataset/Test_Set/text_descriptors/'
test_vis = './data/CoE_dataset/Test_Set/vis_descriptors/'
test_xml = './data/CoE_dataset/Test_Set/XML/'


def dir_contents(input_dir):
    """returns list of files contained in input dir"""

    files = os.listdir(input_dir)
    files.sort()

    # for f in files:
    #     instring = re.split("[.]", f)

    return files


def load_average_audio(path):
    df = pd.read_csv(path, sep=',', header=None)
    df = df.fillna(0)
    avg = df.mean(axis=1)
    return avg


def load_parse_xml(path):
    with open(path, 'rb') as f:
        obj = objectify.fromstring(f.read())
        d = dict(obj['movie'].items())
        return d


def test():
    # load and average audio
    audios = dir_contents(dev_audio)
    df = load_average_audio(dev_audio + audios[0])
    print(df)

    # test loading first xml
    xmls = dir_contents(dev_xml)
    d = load_parse_xml(dev_xml + xmls[0])
    print(d)


if __name__ == "__main__":
    test()


# todo: import
# audio: import files, average each movie per line, use lines as columns/features
# text: do nothing for now
# visual: ??
# XML: parse attributes, filter attributes
