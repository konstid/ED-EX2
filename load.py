import os

from lxml import objectify
import pandas as pd

# dev paths
dev_audio = './data/CoE_dataset/Dev_Set/audio_descriptors/'
dev_audio_avg = './data/CoE_dataset/Dev_Set/audio_avg/'
dev_text = './data/CoE_dataset/Dev_Set/text_descriptors/'
dev_vis = './data/CoE_dataset/Dev_Set/vis_descriptors/'
dev_xml = './data/CoE_dataset/Dev_Set/XML/'

# test paths
test_audio = './data/CoE_dataset/Test_Set/audio_descriptors/'
test_audio_avg = './data/CoE_dataset/Dev_Set/audio_avg/'
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


def load_audio(fdir, fn, avg=False):
    df = pd.read_csv(fdir + fn, sep=',', header=None)
    if avg:
        df = df.fillna(0)
        df = df.mean(axis=1)
    return fn, df


def load_parse_xml(fdir, fn):
    with open(fdir + fn, 'rb') as f:
        obj = objectify.fromstring(f.read())
        d = dict(obj['movie'].items())
        return fn, d


def load_save_all_audios(path=dev_audio):

    audios = dir_contents(path)
    names = []
    dfs = []
    for a in audios:
        name, df = load_audio(path, a, avg=True)

        # save averaged files
        save_path = './data/CoE_dataset/Test_Set/audio_avg/' + name
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path)


def load_all_avg_audios(path=dev_audio_avg):
    audios = dir_contents(path)
    names = []
    dfs = []
    for a in audios:
        name, df = load_audio(path, a)
        names.append(name)
        dfs.append(df)

    audio_df = pd.concat(dfs)
    return audio_df


def load_all_xmls():
    xmls = dir_contents(dev_xml)
    names = []
    dicts = []
    for x in xmls:
        name, d = load_parse_xml(dev_xml, x)
        names.append(name)
        dicts.append(d)

    xml_df = pd.DataFrame(dicts)
    return xml_df


def load_all():
    audio_df = load_all_avg_audios()
    # xml_df = load_all_xmls()
    return


def test():
    # load and average audio
    audios = dir_contents(dev_audio)
    name, df = load_audio(dev_audio, audios[0], avg=True)
    print(df)

    # test loading first xml
    xmls = dir_contents(dev_xml)
    name, d = load_parse_xml(dev_xml, xmls[0])
    print(d)


if __name__ == "__main__":
    # test()
    load_all()


# todo: import
# audio: import files, average each movie per line, use lines as columns/features
# text: do nothing for now
# visual: ??
# XML: parse attributes, filter attributes
