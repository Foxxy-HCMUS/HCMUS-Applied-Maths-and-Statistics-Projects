# from prettytable import PrettyTable
# t = PrettyTable(['Name', 'Age'])
# t.add_row(['Alice', 24])
# t.add_row(['Bob', 19])
# print(t)

from nltk.corpus.reader import TaggedCorpusReader
import glob
from prettytable import PrettyTable
import string


dataset_path = './dataset'
list_of_file = glob.glob(dataset_path + '/*.pos') # Lấy đường dẫn toàn bộ file trong thư mục

reader_corpus = TaggedCorpusReader('.', list_of_file)

list_of_tagged = reader_corpus.tagged_words() # Đọc dữ liệu
# Kết quả sau khi đọc (15 kết quả đầu tiên)

def printList(tList, size = 15):
    t = PrettyTable(['Word', 'Tag'])
    for i in tList[:size]:
        t.add_row([i[0], i[1]])
    print(t)

printList(list_of_tagged)

# Lọc bỏ những từ không có tag (None)
list_of_tagged = list(filter(
    lambda x: x[1] != None, 
    list_of_tagged
))
# Chuyển đổi tag dấu câu thành SYM
list_of_tagged = list(map(
    lambda x: (x[0], "SYM") if (x[0][0] in string.punctuation) else x, 
    list_of_tagged
))
# Lọc tag thừa
list_of_tagged = list(map(
    lambda x: (x[0][:x[0].find("/")], x[1]) if x[0].find("/") != -1 else x,
    list_of_tagged
))

print("Ket Qua sau khi loc: ")
printList(list_of_tagged)