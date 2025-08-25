import docx
import os
from typing import List, Optional

from numpy import ndarray
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm
import pickle

FN_ = 'kk2.docx'
BATCH_SIZE = 12
MAX_STR_LEN = 4096
MAX_RESULTS = 10


def get_model() -> BGEM3FlagModel:
    model = BGEM3FlagModel('BAAI/bge-m3',
                           use_fp16=True)
    return model


def get_max_paragraphs_len(paragraphs: List[str]) -> int:
    return max(len(p) for p in paragraphs)


def calculate_str_len(max_paragraph_len: int) -> int:
    lengths = [1024, 2048, 4096, 8192]
    try:
        return min([i for i in lengths if i >= max_paragraph_len], key=lambda x: abs(x - max_paragraph_len))
    except ValueError:
        return max(lengths)


def filter_too_small_paragraphs(paragraphs: List[str], min_length=25) -> List[str]:
    return [p for p in paragraphs if len(p) > min_length]


def get_paragraphs(filename: str) -> Optional[List[str]]:
    if not os.path.isfile(filename):
        return []
    doc = docx.Document(filename)
    result = []
    for para in doc.paragraphs:
        text = para.text.strip().replace(u'\xa0', ' ')
        if text is not None and text != '':
            result.append(text)

    return result


def make_embeddings(paragraphs: List[str], model: BGEM3FlagModel, progress_bar) -> List[ndarray]:
    return [model.encode(para, batch_size=BATCH_SIZE, max_length=MAX_STR_LEN)['dense_vecs']
            for para in tqdm(paragraphs)]


def calculate_cosine(texts: List[str], embeddings: List[ndarray]):
    with open('output.txt', 'w') as writer:
        for num, em in enumerate(embeddings):
            for num_b, em_b in enumerate(embeddings):
                similarity = em @ em_b.T
                if 0.85 < similarity < 0.9999:
                    writer.write(f'Параграф: \n\n {texts[num]} \n\n очень похож на параграф: \n\n '
                                 f'{texts[num_b]} \n\n с точностью {similarity}'
                                 f' \n\n\n ============================== \n\n\n')

#
# print('loading model')
#
# print('loading texts')
# texts = get_paragraphs(Path(FN_))
# print('calculate lengths')
# lengths = [(num, len(_)) for num, _ in enumerate(texts)]
# max = max([len(_) for _ in texts])
# print(f'max length {max}')
#
# fn_model = Path(FN_ + '.model')
# if not fn_model.is_file():
#     print('making indexes')
#     embeddings = make_embeddings(texts, model)
#     with open(fn_model, 'wb') as handle:
#         pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
# else:
#     print('loading existing indexes')
#     with open(fn_model, 'rb') as handle:
#         embeddings = pickle.load(handle)
#
# # print(len(embeddings[116]))
#
# print('searching similar paragraphs')
# calculate_cosine(texts, embeddings)
# index = create_index(embeddings)
# calculate_similar(texts, embeddings, index)
