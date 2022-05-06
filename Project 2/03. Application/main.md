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

Sau khi tải về từ Kaggle,

Dataset lưu ở `archive/treebank/treebank/combined` và raw data ở `archive/treebank/treebank/raw`, 

Chia thành 2 phần test và train:
- Train: `wsj_0001.mrg` tới `wsj_0190.mrg`
- Test: `wsj_0191` và `wsj_0191.mrg` tới `wsj_0199` và `wsj_0199.mrg`

Theo `treebank/treebank/combined/README`, dữ liệu ban đầu đã được chạy qua PARTS (Ken Church's stochastic part-of-speech tagger), sau đó được sửa lại thông qua người gán nhãn, ghép với câu dữ liệu gốc tạo thành file Bracket. Một số điểm cần phải sửa đổi sau khi load dataset (Các file `.mrg`):
- Các kí hiệu có nhãn dán riêng biệt, để đơn giản cho việc xử lí và thống nhất với các nhãn dán đã liệt kê ở trên, thay đổi nhãn của ký hiệu thành `SYM`
- Một số từ không có nhãn, trong dataset được gán là `-NONE-`, cần lọc ra những từ này trước khi xử lí.
# 3.3 Mô tả các thành phần của mô hình
- Tập các trạng thái ẩn: là tập các tag có thể của mỗi từ (dựa trên quy tắc gán nhãn của Penn TreeBank dataset).

> S = {'JJS', 'PRP$', 'WDT', 'NNP', 'TO', 'PDT', 'WRB', 'WP', 'NNS', 'VB', 'MD', 'RP',  'PRP', 'JJR', 'JJ', 'VBZ', 'RBS', 'VBG', 'POS', 'VBD', 'NN', 'UH', 'FW', 'NNPS', 'WP$', 'EX', 'SYM', 'RBR', 'VBN', 'LS', 'IN', 'DT', 'VBP', 'CD', 'RB', 'CC'}

- Các quan sát có thể: Những từ trong câu theo thứ tự đã được gán nhãn, ví dụ `"I_PRP", "am_VB", "good_JJ",...`
- Các giả thiết của mô hình Markov ẩn phù hợp với tình huống này, do nhãn dán của một từ thường phụ thuộc vào từ phía trước nó (Vd sau động từ khiếm khuyết thường sẽ đi với một động từ nguyên mẫu)

### Các giả thiết được sử dụng
#### 1. Giả thiết của Markov ẩn:
- Xác suất của một trạng thái cụ thể chỉ phụ thuộc vào trạng thái ngay trước đó (Markov Assumptions)
$$P\left({\left. {{q_i}} \right|{q_1} \ldots {q_{i - 1}}} \right) = P\left( {\left. {{q_i}} \right|{q_{i - 1}}} \right)$$
- Xác suất của quan sát đầu ra $o_i$ chỉ phụ thuộc vào trạng thái tạo ra quan sát $q_i$, không bị ảnh hưởng bởi các quan sát và trạng thái khác (Independence Assumption).
$$P\left({\left. {{o_i}} \right|{q_1} \ldots {q_{T}}, {o_1} \ldots {o_{T}}} \right) = P\left( {\left. {{o_i}} \right|{q_{i}}} \right)$$
#### 2. Giả thiết cho POS tagging
- Giả thiết bigram: xác suất của một nhãn chỉ phụ thuộc vào từ phía trước nó, thay vì phụ thuộc vào dãy các nhãn.
$$P\left({{t_1} \ldots {t_{n}}} \right) = \prod_{i=1}^{n}P\left( {\left. {{t_i}} \right|t_0,\ldots ,{t_{i - 1}}} \right)  \approx \prod_{i=1}^{n}P\left( {\left. {{t_i}} \right|{t_{i - 1}}} \right)$$
- Xác xuất một từ xuất hiện dựa trên nhãn độc lập với những từ  và nhãn xung quanh 
$$P\left({{w_1} \ldots {w_{n}}} | {{t_1} \ldots {t_{n}}} \right) \approx \prod_{i=1}^{n}P\left( {\left. {{w_i}} \right|{t_{i}}} \right)$$

### Mục tiêu của bài toán
- Với danh sách từ cho trước $w_1, \ldots, w_n$, ta cần tìm một danh sách nhãn dán $t_1, \ldots, t_n$ phù hợp.
- Nói cách khác, ta cần tìm:
$$\hat{t}_{1:n}=\argmax_{t_1\ldots t_n}{P\left({{t_1} \ldots {t_{n}}} | {{w_1} \ldots {w_{n}}} \right)}$$
Sử dụng định lý Bayes:
$$\hat{t}_{1:n}=\argmax_{t_1\ldots t_n}{\frac{P\left({{w_1} \ldots {w_{n}}} | {{t_1} \ldots {t_{n}}} \right)P\left(t_1\ldots t_n\right)}{P\left(w_1\ldots w_n\right)}}$$
Bỏ qua mẫu số, và áp dụng các giả thuyết đã có, ta được:
$$\hat{t}_{1:n}=\argmax_{t_1\ldots t_n}{\prod_{i=1}^{n}P\left( {\left. {{w_i}} \right|{t_{i}}} \right)P\left( {\left. {{t_i}} \right|{t_{i - 1}}} \right)}$$
Gọi A là ma trận xác suất chuyển từ nhãn này sang nhãn khác, ta tính toán ước lượng hợp lý cực đại (MLE) của xác suất này bằng cách đếm số lượng nhãn thứ 2 theo sau nhãn thứ nhất trên số lượng nhãn thứ nhất:
$$A[t_{i-1}, t_i] = P\left(t_i|t_{i-1}\right) = \frac{C\left(t_{i-1}, t_i\right)}{C\left(t_{i-1}\right)}$$
Gọi B là ma trận xác suất phụ thuộc trạng thái, MLE của xác suất này sẽ là số lần nhãn $t$ được gán cho từ $w$ trên số lượng nhãn $t$:
$$B[t_{i}, w_i] = P\left(w_i|t_{i}\right) = \frac{C\left(t_{i}, w_i\right)}{C\left(t_{i}\right)}$$
Gọi $\pi$ là vector mở đầu phân phối xác suất, được tính bằng số lượng nhãn $t$ mở đầu câu trên tổng số câu:
$$\pi_{t_i} = \frac{C'(t_i)}{C(sentence)}$$
Việc tạo ra được dãy $t_1,\ldots,t_n$ phù hợp với dãy quan sát $o_1,\ldots,o_n$ thông qua việc giải mã, ở đây ta sẽ sử dụng [thuật toán Viterbi](https://en.wikipedia.org/wiki/Viterbi_algorithm)