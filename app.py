import streamlit as st
import numpy as np
import pandas as pd
from fastai2.tabular.all import *
import functools
from pathlib import Path
import sklearn as sk
import math
import altair as alt

#
@st.cache
def prep():
    path = Path('./data')
    df = pd.read_csv(path/'energydata_complete.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['Appliances'] = df['Appliances'].astype(float)
    # df['Appliances'] = (df['Appliances'] - df['Appliances'].mean()) / (df['Appliances'].max() - df['Appliances'].min())
    df['lights'] = df['lights'].astype(float)
    # Split into train, validation and test
    nr = df.shape[0]
    s1 = math.floor(nr * 0.8)
    s2 = math.floor(nr * 0.9)
    data = df[:s2]
    valid_idx = list(range(s1, s2))
    test = df[s2:]
    return data, valid_idx, test

# The return value cannot be hashed by Streamlit
@st.cache(allow_output_mutation=True)
def train(data, valid_idx):
    # Create a learner and train it
    dls = TabularDataLoaders.from_df(data,
      valid_idx=valid_idx,
      bs=64,
      y_names="Appliances",
      cat_names=[],
      cont_names=['T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7',
                  'T8', 'RH_8', 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint'],
      procs=[Categorify, FillMissing, Normalize])

    max_epochs = 200
    learn = tabular_learner(dls, layers=[100, 100, 100, 50, 50], metrics=mse)
    learn.fit_one_cycle(max_epochs)

    return learn

data, valid_idx, test = prep()

with st.spinner("Training..."):
    learn = train(data, valid_idx)
st.success("Trained a model!")

# Predictions
nr_samples = 40
test_cp = test[:nr_samples].copy()
dl = learn.dls.test_dl(test_cp)

# Create a graph
inputs, preds, targets, decoded = learn.get_preds(dl=dl, with_input=True, with_decoded=True, with_loss=False)
test_cp['Appliances_pred'] = preds.numpy()

chart_df = test_cp[['date', 'Appliances', 'Appliances_pred']].melt('date')

# UI
st.sidebar.title("Options")
st.sidebar.markdown("Choose the feature you want to analyse the sensitivity for and within what variation.")
feature = st.sidebar.selectbox('Feature', ['T1', 'T2', 'RH_8', 'RH_2', 'Windspeed'])
variation = st.sidebar.slider('Range', min_value=-5.0, max_value=5.0, value=(-1.0, 1.0), step=0.1)

st.title("Analysis")
st.markdown(f"Choose a point in time to see the sensitivity of the consumed power vs the temperature {feature}. ")
focus_slider = st.slider('Timepoint', value=(nr_samples//2), min_value=0, max_value=nr_samples)
st.text("Predictions")
focus = chart_df.iloc[focus_slider]['date']


def cartesian_product(left, right):
    return (
       left.assign(key=1).merge(right.assign(key=1), on='key').drop('key', 1))


def variations(df, column, steps=np.linspace(-5, 5, num=500)):
    """ Expects a single row"""
    result = df.copy()
    current_value = df[column].iloc[0]
    variations = pd.DataFrame({column: steps+current_value})
    result = result.drop(columns=[column])
    result = cartesian_product(result, variations)
    return result, current_value


sensitivity_df, current_value = variations(test_cp.iloc[[focus_slider]], column=feature, steps=np.linspace(variation[0], variation[1], num=100))
dl = learn.dls.test_dl(sensitivity_df)
preds, targets = learn.get_preds(dl=dl)
sensitivity_df['Appliances_pred'] = preds.numpy()

def sens_chart(sensitivity_df):
    sens = alt.Chart(sensitivity_df).mark_line().encode(
        x=feature,
        y=alt.Y('Appliances_pred'),
        # color=alt.Y('Appliances_pred', scale=alt.Scale(domain=(55,70)))
    )

    focus = alt.Chart(pd.DataFrame({'f': [current_value]})).mark_rule(color='red', strokeWidth=3).encode(
        x="f:Q",
        size=alt.value(3),
        color=alt.ColorValue('red')
    )

    return sens + focus


def charts(focus):

    energy = alt.Chart(chart_df).mark_line(point=True).encode(
        x='date',
        y='value',
        color=alt.Color('variable', legend=alt.Legend(orient='bottom'))
    )

    focus = alt.Chart(pd.DataFrame({'f': [focus]})).mark_rule(color='red', strokeWidth=3).encode(
        x="f:T",
        size=alt.value(3),
        color=alt.ColorValue('red')
    )
    return energy + focus


st.altair_chart(
    charts(focus),
    use_container_width=True
)

st.altair_chart(
    sens_chart(sensitivity_df),
    use_container_width=True
)





