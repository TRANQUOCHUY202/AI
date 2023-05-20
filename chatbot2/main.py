import os
import json
import string
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tkinter as tk
from chatbot import get_response, pred_class, words, classes, lemmatizer

#download các dữ liệu (tokenizer và wordnet) từ thư viện nltk
nltk.download("punkt")
nltk.download("wordnet")

#đọc file intents.json chứa dữ liệu huấn luyện cho chat bot và lưu vào biến data.
dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, 'intents.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)

#khởi tạo biến để lưu trữ các từ, lớp và dữ liệu đầu vào và đầu ra cho huấn luyện mô hình.
words = []
classes = []
data_X = []
data_y = []

#duyệt qua các 'intent' trong data và trích xuất các từ từ mẫu câu hỏi 'patterns'
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        data_X.append(pattern)
        data_y.append(intent["tag"])
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
words = sorted(set(words))
classes = sorted(set(classes))

#tạo ra dữ liệu huấn luyện
training = []
out_empty = [0]* len(classes)

for idx, doc in enumerate(data_X):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)
        output_row= list(out_empty)
        output_row[classes.index(data_y[idx])] = 1

        training.append([bow, output_row])
random.shuffle(training)                #đảm bảo tính ngẫu nhiên trong quá trình huấn luyện.
#chuyển đổi dữ liệu huấn luyện sang dạng numpy array (np.array) để phù hợp với đầu vào và đầu ra của mô hình neural network.
train_X = np.array([i[0] for i in training])
train_Y = np.array([i[1] for i in training])

#xây dựng kiến trúc mô hình neural network bằng framework TensorFlow
model = Sequential()
model.add(Dense(128, input_shape=(len(train_X[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_Y[0]), activation="softmax"))
# tối ưu hóa Adam (adam) với tốc độ học (learning_rate) là 0.01
adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
#compile mô hình với hàm mất mát là categorical crossentropy
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
print(model.summary())
model.fit(x=train_X, y=train_Y, epochs=30, verbose=1) #huấn luyện trên dữ liệu huấn luyện (train_X, train_Y) trong 30 epochs.

lemmatizer = WordNetLemmatizer()
#định nghĩa các hàm clean_text và bag_of_words để tiền xử lý 
def clean_text(text):
    tokens = nltk.word_tokenize(text) 
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens
def bag_of_words(text, vocab):
    tokens = clean_text(text)

    bow = [0] * len(vocab)

    for w in tokens:

        for idx, word in enumerate(vocab):

            if word == w:

                bow[idx] = 1

    return np.array(bow)
#định nghĩa các hàm pred_class và get_response 
#dự đoán lớp của câu hỏi dựa trên mô hình
#và trả về câu trả lời phù hợp từ dữ liệu huấn luyện.
def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    result = model.predict(np.array([bow]))[0] 
    thresh = 0.5
    y_pred = [[indx, res] for indx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]]) 
    return return_list
def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
     result = "Sorry! I don't understand."
    else:
        tag = intents_list[0]
        list_of_intents = intents_json["intents"]
        for i in list_of_intents:
            if i["tag"] == tag:
                result = random.choice(i["responses"])
                break

    return result

#định nghĩa hàm send để xử lý sự kiện khi người dùng gửi câu hỏi.
def send():
    message = entry_box.get()
    entry_box.delete(0, tk.END)

    if message == '0':
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, "You: " + message + "\n\n")
        chat_log.config(foreground="#442265", font=("Verdana", 12))
        chat_log.insert(tk.END, "ChatBot: Goodbye! Have a nice day.\n\n")
        chat_log.config(state=tk.DISABLED)
        chat_log.yview(tk.END)
    else:
        intents = pred_class(message, words, classes)
        result = get_response(intents, data)
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, "You: " + message + "\n\n")
        chat_log.config(foreground="#442265", font=("Verdana", 12))
        chat_log.insert(tk.END, "ChatBot: " + result + "\n\n")
        chat_log.config(state=tk.DISABLED)
        chat_log.yview(tk.END)

#tạo giao diện người dùng cho chat bot bằng thư viện tkinter
root = tk.Tk()
root.title("Chatbot")

chat_log = tk.Text(root, bd=0, bg="white", height="8", width="50", font="Arial")
chat_log.config(state=tk.DISABLED)

scrollbar = tk.Scrollbar(root, command=chat_log.yview)
chat_log['yscrollcommand'] = scrollbar.set

entry_box = tk.Entry(root, bd=0, bg="white", width="29", font="Arial")
entry_box.bind("<Return>", send)

send_button = tk.Button(root, text="Send", command=send)

# Đặt tất cả các thành phần trên màn hình
scrollbar.place(x=710,y=6, height=386)
chat_log.place(x=25,y=6, height=386, width=700)
entry_box.place(x=25, y=401, height=30, width=680)
send_button.place(x=690, y=401, height=30)

root.mainloop()

print("Enter 0 if you want to exit the ChatBot.")
while True:
    message = input("You: ")
    if message == "0":
        break

    intents = pred_class(message, words, classes) 
    result = get_response(intents, data)

    print("ChatBot:", result)


