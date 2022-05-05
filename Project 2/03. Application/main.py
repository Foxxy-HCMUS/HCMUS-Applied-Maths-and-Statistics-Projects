from nltk.corpus.reader import BracketParseCorpusReader
import glob
import string


dataset_path = './dataset'
list_of_file = glob.glob(dataset_path + '/train/' + '/*.mrg') # Lấy đường dẫn toàn bộ file trong thư mục

reader_corpus = BracketParseCorpusReader('.', list_of_file)

# Đọc dữ liệu các câu
list_of_tagged_sents = reader_corpus.tagged_sents() 

def printSentence(tag_sent, word_per_line = 2):
    word_per_line = min(word_per_line, len(tag_sent) - 1)
    print('[', end='')
    for i in range(0, len(tag_sent)):
        if i % word_per_line == 0:
            print('')
            print('\t', end = '')
        print(tag_sent[i], end = ', ')
    print('\n]')


print("Du lieu ban dau:")
printSentence(list_of_tagged_sents[0])

# Lọc bỏ những từ không có tag (-NONE-)
list_of_tagged_sents = list(map(
    lambda sent: list(filter(
        lambda word: word[1] != '-NONE-',
        sent
    )),
    list_of_tagged_sents
))

# Chuyển đổi tag dấu câu thành SYM
list_of_tagged_sents = list(map(
    lambda sent: list(map(
        lambda word: (word[0], "SYM") if word[1][0] in string.punctuation else word,
        sent
    )),
    list_of_tagged_sents
))
print("Ket Qua sau khi loc: ")
printSentence(list_of_tagged_sents[0])

# Vd bigram: [1,2,3,4] -> [(1,2), (2,3), (3,4)]
# P(t_i | t_{i - 1})
def bigramCount(tag_sent_list):
    # Tạo bigram trên tag
    # Bigram được tạo riêng trên mỗi câu!

    bigram = [
        (sent[i][1], sent[i + 1][1]) for sent in tag_sent_list for i in range(len(sent) - 1)
    ]

    map_count = {}
    for x in bigram:
        if x in map_count:
            map_count[x] += 1
        else:
            map_count[x] = 1
    return map_count

# Tương tự với unigram
def unigramCount(tag_sent_list):
    unigram = [
        word[1] for sent in tag_sent_list for word in sent
    ]
    map_count = {}
    for x in unigram:
        if x in map_count:
            map_count[x] += 1
        else:
            map_count[x] = 1
    return map_count

# Tagset (36 loại)
tag_set = [
    'JJS', 'PRP$', 'WDT', 'NNP', 'TO', 'PDT', 'WRB', 'WP', 'NNS', 'VB', 'MD', 'RP', 
    'PRP', 'JJR', 'JJ', 'VBZ', 'RBS', 'VBG', 'POS', 'VBD', 'NN', 'UH', 'FW', 'NNPS', 
    'WP$', 'EX', 'SYM', 'RBR', 'VBN', 'LS', 'IN', 'DT', 'VBP', 'CD', 'RB', 'CC'
]

bi_cnt = bigramCount(list_of_tagged_sents)
uni_cnt = unigramCount(list_of_tagged_sents)

print("5 bigram có số lần xuất hiện lớn nhất: ")
for i in sorted(bi_cnt, key = bi_cnt.get)[-5:]:
    print((i, bi_cnt[i]))

print("5 unigram  có số lần xuất hiện lớn nhất: ")
for i in sorted(uni_cnt, key = uni_cnt.get)[-5:]:
    print((i, uni_cnt[i]))