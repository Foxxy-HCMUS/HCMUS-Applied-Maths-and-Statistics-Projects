# 3. Vận dụng
Ứng dụng của mô hình Markov ẩn được lựa chọn: _**`POS tagging`**_ (gán nhãn ngữ pháp)

## 3.1 Mô tả bài toán
- Đầu vào: Mảng các từ trong một câu theo thứ tự

    Ví dụ: `input = ["the", "Dutch", "publishing", "group"]`
- Đầu ra kỳ vọng: mảng các từ cùng với nhãn đã được gán (sau khi tính toán). Một số từ tiếng anh có nhiều dạng (ví dụ theo từ điển, một từ vừa là *danh từ*, vừa là *động từ* ...) cần phải xác định vai trò của từ trong câu để có thể xác định chính xác nghĩa của từ đó.

    Ví dụ: 
    ```python
    output = [
        ("the", "DT"),
        ("Dutch", "NNP"),
        ("publishing", "VBG"),
        ("group", "NN"),
    ]
    ```

## 3.2 Tập dữ liệu & Các bước tiền xử lý
### 3.2.1 Tập dữ liệu
Tập dữ liệu: Trích xuất từ [Penn TreeBank](https://www.kaggle.com/datasets/nltkdata/penn-tree-bank) (~5%)

Các nhãn gán được sử dụng (Penn TreeBank):

Nguồn: [Speech and Language Processing. Daniel Jurafsky & James H. Martin.](https://web.stanford.edu/~jurafsky/slp3/8.pdf)

|Nhãn|Ý nghĩa|Ví dụ|
|---|---|---|
|CC| Liên từ kết hợp | and, but, or |
|CD| Số đếm | one, two |
|DT| Định từ | a, the |
|EX| Tồn tại _there_ | there |
|FW| Từ mượn | mea culpa |
|IN| Giới từ | of, in, by |
|JJ| Tính từ | yellow |
|JJR| Tính từ so sánh hơn | bigger |
|JJS| Tính từ so sánh nhất | wildest |
|LS| Đánh dấu danh sách | 1, 2, One |
|MD| Động từ khiếm khuyết | can, should |
|NN| Danh từ số ít | llama |
|NNS| Danh từ số nhiều | llamas|
|NNP| Danh từ riêng số ít | IBM |
|NNPS| Danh từ riêng số nhiều | Carolinas |
|PDT| Từ chỉ định | all, both |
|POS| Kết thúc sở hữu cách | 's |
|PRP| Đại từ nhân xưng | I, you, he |
|PRP$| Đại từ sở hữu | your, one's |
|RB| Trạng từ | quickly |
|RBR| Trạng từ so sánh hơn | faster |
|RBS| Trạng từ so sánh nhất | fastest |
|RP| Tiểu từ | up, off |
|SYM| Ký hiệu | +, %, & |
|TO| Từ _to_ | to |
|UH| Thán từ | ah, oops |
|VB| Động từ nguyên mẫu | eat |
|VBD| Động từ quá khứ | ate |
|VBG| Danh động từ | eating |
|VBN| Động từ quá khứ phân từ | eaten |
|VBP| Động từ ngôi thứ 3 số ít | eat |
|VBZ| Động từ ngôi thứ 3 số nhiều | eats |
|WDT| Wh- xác định | which, that |
|WP| Wh- đại từ | what, who |
|WP$| Wh- sỡ hữu | whose |
|WRB| Wh- trạng từ | how, where|

### 3.2.2 Tiền xử lí
Sử dụng thư viện nltk để đọc dataset,
Dataset lưu ở `./treebank/treebank/tagged`, copy toàn bộ file `*.pos` tới thư mục dataset ở nơi chứa mã nguồn.
```bash
# Giả sử file sau khi tải về ở kaggle  và giải nén lưu ở ~/
cp ~/archive/treebank/treebank/tagged/*.pos ./dataset/
```

Theo `treebank/treebank/tagged/README`, dữ liệu ban đầu đã được chạy qua PARTS (Ken Church's stochastic part-of-speech tagger), sau đó được sửa lại thông qua người gán nhãn. Một số điểm cần phải sửa đổi sau khi load dataset (Các file `.pos`):
- Dấu ngoặc vuông ( **\[ \]** ) được tạo với PARTS là dư thừa, bỏ qua trong quá trình xử lí
- Các kí hiệu có nhãn dán riêng biệt, để đơn giản cho việc xử lí và thống nhất với các nhãn dán đã liệt kê ở trên, thay đổi nhãn của ký hiệu thành `SYM`
- Một số từ không chắc chắn thì sẽ có nhiều hơn 1 nhãn. Tuy nhiên, `TaggedCorpusReader` từ thư viện `nltk.corpus.reader` chỉ có thể đọc 1 nhãn (cuối cùng). Ví dụ `"A/NNP/FW"` sẽ được đọc thành `("A/NNP", "FW")`, ta cần xử lý phần word (loại bỏ tag dư) để trở thành `("A", "FW")`