import os.path

import sklearn
import streamlit as st
import numpy as np
import pandas as pd
import glob
import pickle

from sklearn import compose


def prepare_data(data):
    data['Cabin number'] = data.Cabin.map(lambda x: str(x)[1:].strip())
    data['Cabin number'] = data['Cabin number'].replace('an', '0').astype(int, copy=False)
    data['Cabin'] = data.Cabin.map(lambda x: str(x)[0].strip())

    data['Ticket'] = data.Ticket.fillna('X')
    data['Embarked'] = data.Embarked.fillna('X')

    return pd.get_dummies(data, columns=['Sex', 'Cabin', 'Embarked'], prefix=['sex', 'cab', 'emb'])


def page_description():
    st.title("[Tabular playgrounds - April 2021 Model](https://www.kaggle.com/c/tabular-playground-series-apr-2021/)")
    st.write(
        "Данное соревнование организовывается каждый месяц с целью помочь новичкам освоить основы машинного обучения "
        "и участия в соревнованиях на платформе Kaggle.\nВ этом месяце данные соревнования сгенерированы с помощью "
        "CTGAN на основе набора данных «Титаник».\nЗадача в данном соревновании – определить, выжил ли пассажир "
        "корабля «Synthanic» на основе следующих признаков:\n- pclass – класс билета пассажира;\n- sex – пол "
        "пассажира;\n- age – возраст пассажира;\n- sibsp – количество братьев, сестёр и супругов на борту "
        "корабля;\n- parch – количество детей и родителей на борту корабля;\nticket – номер билета пассажира;\n- fare "
        "– "
        "стоимость билета;\n- cabin – номер каюты пассажира;\n- embarked – порт, в котором пассажир поднялся на борт "
        "корабля.")


@st.cache
def load_data():
    return pd.read_csv("competition\\data\\train.csv")


def page_data():
    st.write("# Данные")
    df = load_data()
    st.write(df.head(100))
    st.write("\n# Графики")
    for filepath in glob.iglob(r'competition\plots\*.png'):
        st.image(filepath)


def page_predict():
    # PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
    file = st.file_uploader('Загрузить csv файл с предобработанными данными', 'csv')
    submit = st.button("Предсказать")
    if file is not None and submit:
        with open('competition\\modules\\saved_model.pkl', 'rb') as pkl_file:
            model = pickle.load(pkl_file)
        data = pd.read_csv(file).drop(columns=['Ticket', 'Name', 'PassengerId'])
        st.write(data)
        y_pred = model.predict(data)
        st.write(pd.DataFrame({'y_pred': y_pred}))


page_predict()
