import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import json
from custom.graph_embedding import Graph2Vec
from sklearn.metrics.pairwise import cosine_similarity


def get_similarity(source_embedding, target_embeddings):
    result = []
    # print(target_embeddings)
    for target in target_embeddings:
        # similarity = np.linalg.norm(target - source_embedding)
        similarity = np.dot(target, source_embedding) / (
            np.linalg.norm(target) * np.linalg.norm(source_embedding)
        )
        result.append(similarity)
    return result


def json_to_numpy(x):
    return np.array(json.loads(x))


def get_reference_dataset(filepath):
    df = pd.read_csv(filepath)
    df["time"] = df.time.apply(json_to_numpy)
    df["machine"] = df.machine.apply(json_to_numpy)
    df["embedding"] = df.embedding.apply(json_to_numpy)

    return df


def embedding_by_graphvec(graphs):
    # graph = make_graph(time_matrix, machine_matrix)
    model = Graph2Vec(use_node_attribute="machine", dimensions=128, wl_iterations=30)
    model.fit(graphs)
    embeddings = model.get_embedding()
    return embeddings


def get_embedding(time_matrix, machine_matrix):
    graph = make_graph(time_matrix, machine_matrix)
    model = Graph2Vec(use_node_attribute="machine", dimensions=128, wl_iterations=30)
    model.fit([graph])
    embedding = model.get_embedding()
    return embedding[0]


def generate_instance(num_jobs, num_machines, time_min=1, time_max=100, random_seed=0):

    np.random.seed(random_seed)  # Seed for reproducibility

    # Initialize arrays
    times = np.zeros((num_jobs, num_machines), dtype=int)
    machines = np.zeros((num_jobs, num_machines), dtype=int)

    for i in range(num_jobs):
        # Generate non-duplicate times for each job
        times[i] = np.random.choice(
            range(time_min, time_max + 1), size=num_machines, replace=False
        )
        # Generate non-duplicate machines for each job
        machines[i] = np.random.permutation(range(1, num_machines + 1))

    return times, machines


def print_instance(times, machines):
    num_jobs, num_machines = times.shape

    print("Times")
    for j in range(num_jobs):
        for m in range(num_machines):
            print(f"{times[j, m]:3d}", end=" ")
        print()

    print("\nMachines")
    for j in range(num_jobs):
        for m in range(num_machines):
            print(f"{machines[j, m]:3d}", end=" ")
        print()


def make_graph(processing_time_matrix, machine_matrix) -> nx.DiGraph:
    G = nx.DiGraph()

    # Conjunctive Graph
    for job_index, (time_row, machine_row) in enumerate(
        zip(processing_time_matrix, machine_matrix)
    ):
        previous_node = None
        for step_index, (time, machine) in enumerate(zip(time_row, machine_row)):
            # 노드 추가
            node = f"{job_index}-{step_index}"
            G.add_node(node, machine=machine, time=time)

            # Conjunctive Graph (동일 작업 내)
            if previous_node:
                G.add_edge(previous_node, node, type="CONJUNCTIVE")
            previous_node = node

    # Disjunctive Graph (동일 기계 사용)
    machine_indexes = set(machine_matrix.flatten().tolist())
    for m_idx in machine_indexes:
        job_ids, step_ids = np.where(machine_matrix == m_idx)

        for job_id, step_id in zip(job_ids, step_ids):
            node = f"{job_id}-{step_id}"
            for job_id2, step_id2 in zip(job_ids, step_ids):
                if not (job_id == job_id2 and step_id == step_id2):
                    other_node = f"{job_id2}-{step_id2}"
                    G.add_edge(other_node, node, type="DISJUNCTIVE")

    return G


def draw_graph(G: nx.DiGraph, machine_num: int, node_size=10):
    # 노드 색상 배열 생성
    node_colors = [G.nodes[node]["machine"] for node in G.nodes]

    pos = nx.spring_layout(G)

    edge_x_conjunctive = []
    edge_y_conjunctive = []
    edge_x_disjunctive = []
    edge_y_disjunctive = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        if edge[2]["type"] == "CONJUNCTIVE":
            edge_x_conjunctive.append(x0)
            edge_x_conjunctive.append(x1)
            edge_x_conjunctive.append(None)
            edge_y_conjunctive.append(y0)
            edge_y_conjunctive.append(y1)
            edge_y_conjunctive.append(None)
        elif edge[2]["type"] == "DISJUNCTIVE":
            edge_x_disjunctive.append(x0)
            edge_x_disjunctive.append(x1)
            edge_x_disjunctive.append(None)
            edge_y_disjunctive.append(y0)
            edge_y_disjunctive.append(y1)
            edge_y_disjunctive.append(None)

    edge_trace_conjunctive = go.Scatter(
        x=edge_x_conjunctive,
        y=edge_y_conjunctive,
        line=dict(width=0.5, color="blue"),
        hoverinfo="none",
        mode="lines",
        name="Conjunctive",
    )

    edge_trace_disjunctive = go.Scatter(
        x=edge_x_disjunctive,
        y=edge_y_disjunctive,
        line=dict(width=0.5, color="red"),
        hoverinfo="none",
        mode="lines",
        name="Disjunctive",
    )

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            color=node_colors,
            colorscale="Viridis",
            size=node_size,
            colorbar=dict(
                thickness=15,
                title="Machine",
                xanchor="left",
                titleside="right",
            ),
            line_width=2,
        ),
    )

    node_adjacencies = []
    # node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        # node_text.append(f'Machine {G.nodes[node]["machine"]}')

    fig = go.Figure(
        data=[edge_trace_conjunctive, edge_trace_disjunctive, node_trace],
        layout=go.Layout(
            title="Job Shop Scheduling Disjunctive Graph",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=10, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Conjunctive: Blue, Disjunctive: Red",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
        ),
    )

    return fig


def random_mask(
    processing_time_matrix,
    machine_matrix,
    num_job,  # job 개수
    num_machine,  # machine 개수
    decrement_num_job,  # 축소시킬 job 사이즈
    decrement_num_machine,  # 각 job의 축소시킬 operation 사이즈
):
    random_jobs = np.sort(
        np.random.choice(range(num_job), decrement_num_job, replace=False)
    )

    deprecated_time_matrix = processing_time_matrix[random_jobs]
    deprecated_machine_matrix = machine_matrix[random_jobs]

    start_machine = num_machine - decrement_num_machine
    deprecated_time_matrix = deprecated_time_matrix[:, start_machine:]
    deprecated_machine_matrix = deprecated_machine_matrix[:, start_machine:]

    return deprecated_time_matrix, deprecated_machine_matrix


def return_scheduling(num_jobs=100, num_machines=20, schedule_cycle=10):
    env = HeuristicAttentionJsspEnv(
        num_jobs=num_jobs, num_machines=num_machines, schedule_cycle=schedule_cycle
    )
    env.reset()
    for _ in tqdm(range(700)):
        env.step(env.action_space.sample())

    fig_gannt = env.render()
    return fig_gannt
