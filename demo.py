import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import dash
from dash import Dash, html, dcc, dash_table, Input, Output, State

import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import time
import threading

from custom.utils import (
    generate_instance,
    print_instance,
    make_graph,
    draw_graph,
    get_reference_dataset,
    embedding_by_graphvec,
    get_similarity,
    get_embedding,
    return_scheduling,
)
from custom.graph_embedding import node2vector

sys.path.extend("./")
from gymjsp.jsspenv import HeuristicAttentionJsspEnv
from gymjsp.operationControl import JobSet
from gymjsp.orliberty import load_instance

app = Dash(__name__)
app.layout = html.Div(
    children=[
        html.H1(children="스마트팩토리캡스톤디자인 6조 프로젝트 Demo"),
        html.H3(children="Random Seed 입력"),
        dcc.Input(
            id="input-random-seed",
            type="text",
            style={"margin": "10px 20px", "padding": "10px"},
        ),
        html.Button(
            "Submit",
            id="submit-button",
            style={
                "background-color": "blue",  # 버튼 배경색
                "border": "none",  # 테두리 제거
                "color": "white",  # 글자 색
                "padding": "15px 20px",  # 패딩
                "text-align": "center",  # 텍스트 정렬
                "text-decoration": "none",  # 텍스트 장식 제거
                "display": "inline-block",  # 인라인 블록
                "font-size": "16px",  # 글자 크기
                "margin": "10px 20px",  # 마진
                "cursor": "pointer",  # 커서 포인터
                "display": "inline-block",  # 인라인 블록
                "border-radius": "10px",  # 테두리 모서리
            },
        ),
        dcc.Store(id="progress-data", data=0),
        dcc.Interval(id="interval-progress", interval=500, n_intervals=0),
        html.Div(
            id="content-graph",
            style={"display": "none"},
            children=[
                html.H2(children="분리형 그래프(Disjunctive Graph) 시각화"),
                html.Div(children="분리형 그래프 (Machine 10개, Job 10개)"),
                dcc.Graph(id="graph_test"),
                html.Div(children="분리형 그래프 (Machine 20개, Job 100개)"),
                dcc.Graph(id="graph_real"),
                html.H2(children="유사도 분석 결과"),
                dcc.Graph(id="similarity"),
                html.H3(children="적합한 p, q 값"),
                html.Div(id="pq-values", children=""),
            ],
        ),
        html.Div(
            id="content-embedding",
            style={"display": "none"},
            children=[
                html.H2(children="노드 임베딩 결과"),
                html.P("Faster Node2Vec을 이용한 노드 임베딩 결과 (2000x20)"),
                html.Div(id="embedding-status", children="임베딩 진행 중..."),
                dash_table.DataTable(
                    id="table",
                    page_size=20,  # 한 페이지에 보여줄 행의 수
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "height": "auto",
                        "minWidth": "50px",
                        "width": "50px",
                        "maxWidth": "50px",
                        "whiteSpace": "normal",
                    },
                ),
                html.Button(
                    "Scheduling Start",
                    id="scheduling-button",
                    style={
                        "background-color": "blue",  # 버튼 배경색
                        "border": "none",  # 테두리 제거
                        "color": "white",  # 글자 색
                        "padding": "15px 20px",  # 패딩
                        "text-align": "center",  # 텍스트 정렬
                        "text-decoration": "none",  # 텍스트 장식 제거
                        "display": "inline-block",  # 인라인 블록
                        "font-size": "16px",  # 글자 크기
                        "margin": "10px 20px",  # 마진
                        "cursor": "pointer",  # 커서 포인터
                        "display": "inline-block",  # 인라인 블록
                        "border-radius": "10px",  # 테두리 모서리
                    },
                ),
            ],
        ),
        html.Div(
            id="content-gannt-chart",
            style={"display": "none"},
            children=[
                html.H2(children="최종 스케줄링 Gannt Chart 결과"),
                dcc.Graph(id="gannt_chart"),
                dcc.Interval(id="interval-component", interval=1 * 1000, n_intervals=0),
            ],
        ),
        # dcc.Progress(id="pregress-bar", value=0, max=100),
    ]
)


@app.callback(
    [
        Output("content-graph", "style"),
        Output("graph_test", "figure"),
        Output("graph_real", "figure"),
        Output("similarity", "figure"),
        Output("pq-values", "children"),
    ],
    [Input("submit-button", "n_clicks"), Input("input-random-seed", "value")],
)
def update_after_random_seed(n_clicks, seed_value):
    if n_clicks is None:
        return dash.no_update

    seed = int(seed_value)
    print(seed)

    # Generate instance
    processing_time_matrix, machine_matrix = generate_instance(10, 10, random_seed=seed)
    global G_10
    G_10 = make_graph(processing_time_matrix, machine_matrix)

    processing_time_matrix2, machine_matrix2 = generate_instance(
        100, 20, random_seed=seed
    )
    global G_100
    G_100 = make_graph(processing_time_matrix2, machine_matrix2)

    df = get_reference_dataset("store_pq.csv")
    embeddings = df.embedding.values
    new_embedding = get_embedding(processing_time_matrix2, machine_matrix2)
    similarities = get_similarity(new_embedding, embeddings)

    scaler = MinMaxScaler()
    similarities = scaler.fit_transform(np.array(similarities).reshape(-1, 1)).flatten()
    max_similarity_index = np.argmax(similarities)
    colors = [
        "red" if i == max_similarity_index else "blue" for i in range(len(similarities))
    ]
    fig_similarity = go.Figure(
        go.Bar(
            x=[f"Graph {i+1}" for i in range(9)], y=similarities, marker_color=colors
        )
    )
    p_appropriate = df.iloc[max_similarity_index].p
    q_appropriate = df.iloc[max_similarity_index].q

    fig_test = draw_graph(G_10, 10, 15)
    fig_real = draw_graph(G_100, 20, 8)

    return (
        {"display": "block"},
        fig_test,
        fig_real,
        fig_similarity,
        f"p: {p_appropriate}, q: {q_appropriate}",
    )


@app.callback(
    [
        Output("content-embedding", "style"),
        Output("embedding-status", "children"),
        Output("table", "columns"),
        Output("table", "data"),
    ],
    Input("content-graph", "style"),
)
def update_embedding(style):
    if style["display"] == "none":
        return dash.no_update

    # Embedding by nodevectors
    global G_100
    embedding = node2vector(G_100, type="faster", dimensions=20)
    df_embedding = pd.DataFrame(embedding, columns=[f"E{i}" for i in range(20)])

    return (
        {"display": "block"},
        "임베딩 완료",
        [{"name": i, "id": i} for i in df_embedding.columns],
        df_embedding.to_dict("records"),
    )


@app.callback(
    [
        Output("content-gannt-chart", "style"),
        Output("gannt_chart", "figure"),
    ],
    Input("scheduling-button", "n_clicks"),
)
def start_scheduling(n_clicks):
    if n_clicks is None:
        return dash.no_update

    fig_gannt = return_scheduling(num_jobs=100, num_machines=20, schedule_cycle=10)

    return (
        {"display": "block"},
        fig_gannt,
    )


if __name__ == "__main__":
    app.run_server(debug=False)
