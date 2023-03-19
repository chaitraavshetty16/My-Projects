'''
This code is to create UI for seq2seq model using streamlit
'''


import streamlit as st
from streamlit_chat import message
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.cost import cross_entropy_seq, cross_entropy_seq_with_mask
from tqdm import tqdm
from sklearn.utils import shuffle
from data.twitter import data
from tensorlayer.models.seq2seq import Seq2seq
from tensorlayer.models.seq2seq_with_attention import Seq2seqLuongAttention
import os




def initial_setup(data_corpus):
    metadata, idx_q, idx_a = data.load_data(PATH='data/{}/'.format(data_corpus))
    (trainX, trainY), (testX, testY), (validX, validY) = data.split_dataset(idx_q, idx_a)
    trainX = tl.prepro.remove_pad_sequences(trainX.tolist())
    trainY = tl.prepro.remove_pad_sequences(trainY.tolist())
    testX = tl.prepro.remove_pad_sequences(testX.tolist())
    testY = tl.prepro.remove_pad_sequences(testY.tolist())
    validX = tl.prepro.remove_pad_sequences(validX.tolist())
    validY = tl.prepro.remove_pad_sequences(validY.tolist())
    return metadata, trainX, trainY, testX, testY, validX, validY




class seq2seq_model():
    def __init__(self) -> None:
        data_corpus = "cornell_corpus"
        #data preprocessing
        metadata, trainX, trainY, testX, testY, validX, validY = initial_setup(data_corpus)

        batch_size = 32
        src_vocab_size = len(metadata['idx2w']) # 8002 (0~8001)
        emb_dim = 1024

        word2idx = metadata['w2idx']   # dict  word 2 index
        idx2word = metadata['idx2w']   # list index 2 word

        unk_id = word2idx['unk']   # 1
        pad_id = word2idx['_']     # 0

        start_id = src_vocab_size  # 8002
        end_id = src_vocab_size + 1  # 8003

        word2idx.update({'start_id': start_id})
        word2idx.update({'end_id': end_id})
        idx2word = idx2word + ['start_id', 'end_id']

        src_vocab_size = tgt_vocab_size = src_vocab_size + 2

        self.word2idx = word2idx
        self.idx2word = idx2word
        self.start_id = start_id
        self.end_id = end_id
        self.unk_id = unk_id

        vocabulary_size = src_vocab_size
        decoder_seq_length = 20

        # create model object
        model_ = Seq2seq(
                decoder_seq_length = decoder_seq_length,
                cell_enc=tf.keras.layers.GRUCell,
                cell_dec=tf.keras.layers.GRUCell,
                n_layer=3,
                n_units=256,
                embedding_layer=tl.layers.Embedding(vocabulary_size=vocabulary_size, embedding_size=emb_dim),
                )
            
        optimizer = tf.optimizers.Adam(learning_rate=0.001)

        # Load the pretrained model
        load_weights = tl.files.load_npz(name='model-cornell_corpus.npz')
        tl.files.assign_weights(load_weights, model_)

        self.model = model_

    def inference(self,seed, top_n):
        self.model.eval()
        seed_id = [self.word2idx.get(w, self.unk_id) for w in seed.split(" ")]
        sentence_id = self.model(inputs=[[seed_id]], seq_length=20, start_token=self.start_id, top_n = top_n)
        sentence = []
        for w_id in sentence_id[0]:
            w = self.idx2word[w_id]
            if w == 'end_id':
                break
            sentence = sentence + [w]
        return sentence

    def predict(self,question,top_n):
        answers = []
        for i in range(top_n):
            answers.append(self.inference(question, top_n))
            
        return answers


seq2seq_model = seq2seq_model()

st.title("This is demo to test seq2seq model trained on Cornell Corpus Data")
st.subheader("This model will reply to your query based on the cornell corpus movie data")

#st.balloons()


# with st.form(key='my_form'):
#     question = st.text_input("Please post your Question here.")
#     submit = st.form_submit_button('Submit')

#     if submit:
#         answer = seq2seq_model.inference(question,3)
        
#         st.write(" >", ' '.join(answer))

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def get_text():
    input_text = st.text_input("You: ","you can do it", key="input")
    return input_text 


user_input = get_text()

if user_input:
    answer = seq2seq_model.inference(user_input,3)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(' '.join(answer))


if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', )
        

