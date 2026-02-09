#!/usr/bin/env python3
"""
Lerobotデータセットのparquetファイルから、指定されたエピソードの
observation.stateとactionをtimestampに対して散布図としてプロットするスクリプト。

Usage:
    uv run python plot.py --parquet PATH_TO_PARQUET --episode EPISODE_INDEX [--save OUTPUT_PATH]
"""


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

DIM = 14  # dimention of "action" and "observation.state"
DEFAULT_PARQUET_PATH = Path(
    "/Users/nao.yamada/personal/lerobot-playground/datasets/0115_1-75_converted_0116_1-116_converted_single_arm_7dim/data/chunk-000/file-000.parquet"
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot observation.state and action as scatter plots "
            "against timestamp for a single episode."
        )
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=DEFAULT_PARQUET_PATH,
        help=f"Path to parquet file (default: {DEFAULT_PARQUET_PATH})",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Episode index to plot. If omitted, the first episode in the file is used.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the figure (e.g., outputs/plot.html).",
    )
    return parser.parse_args()


def pick_episode(df: pd.DataFrame, episode: int | None) -> int:
    if episode is not None:
        return episode
    unique_eps = df["episode_index"].dropna().unique()
    if len(unique_eps) == 0:
        raise ValueError("No episode_index values found in the parquet file.")
    return int(sorted(unique_eps)[0])


def load_vector_column(df: pd.DataFrame, column: str, expected_dim: int = DIM) -> pd.DataFrame:
    values = df[column].to_list()
    vectors = pd.DataFrame(values)
    if vectors.shape[1] != expected_dim:
        raise ValueError(
            f"{column} should have {expected_dim} elements, got {vectors.shape[1]}."
        )
    return vectors


def calculate_tick_interval(data_range: float) -> float:
    """
    データの範囲に基づいて適切な目盛り間隔を計算する。
    細かすぎず、荒すぎない間隔を自動設定する。
    """
    if data_range == 0:
        return 0.01

    # データ範囲の桁数を取得
    magnitude = 10 ** np.floor(np.log10(data_range))
    normalized_range = data_range / magnitude

    # 正規化された範囲に応じて基本の目盛り間隔を決定
    if normalized_range <= 1.5:
        tick_interval = 0.1 * magnitude
    elif normalized_range <= 3:
        tick_interval = 0.2 * magnitude
    elif normalized_range <= 7:
        tick_interval = 0.5 * magnitude
    else:
        tick_interval = 1 * magnitude

    return tick_interval


def main() -> None:
    args = parse_args()
    parquet_path = args.parquet
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    episode = pick_episode(df, args.episode)
    df = df[df["episode_index"] == episode].copy()
    if df.empty:
        raise ValueError(f"No rows found for episode_index={episode}.")

    timestamps = df["timestamp"]
    obs = load_vector_column(df, "observation.state")
    act = load_vector_column(df, "action")

    # 各軸の範囲を計算して目盛り間隔を設定
    timestamp_range = timestamps.max() - timestamps.min()
    timestamp_tick = calculate_tick_interval(timestamp_range)

    obs_min, obs_max = obs.min().min(), obs.max().max()
    obs_range = obs_max - obs_min
    obs_tick = calculate_tick_interval(obs_range)

    act_min, act_max = act.min().min(), act.max().max()
    act_range = act_max - act_min
    act_tick = calculate_tick_interval(act_range)

    # Observation State グラフ
    fig_obs = go.Figure()
    for idx in range(obs.shape[1]):
        fig_obs.add_trace(
            go.Scattergl(
                x=timestamps,
                y=obs[idx],
                mode="markers",
                marker=dict(size=3),
                name=f"state[{idx}]",
                showlegend=True,
            )
        )
    fig_obs.update_xaxes(title_text="timestamp", dtick=timestamp_tick)
    fig_obs.update_yaxes(title_text="observation.state value", dtick=obs_tick)
    fig_obs.update_layout(
        title=f"Observation State (episode {episode})",
        height=800,
        width=2000,
        legend_title_text="Series",
    )

    # Action グラフ
    fig_act = go.Figure()
    for idx in range(act.shape[1]):
        fig_act.add_trace(
            go.Scattergl(
                x=timestamps,
                y=act[idx],
                mode="markers",
                marker=dict(size=3),
                name=f"action[{idx}]",
                showlegend=True,
            )
        )
    fig_act.update_xaxes(title_text="timestamp", dtick=timestamp_tick)
    fig_act.update_yaxes(title_text="action value", dtick=act_tick)
    fig_act.update_layout(
        title=f"Action (episode {episode})",
        height=800,
        width=2000,
        legend_title_text="Series",
    )

    # 2つのグラフを1つのHTMLに結合
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        # HTMLを結合して保存
        with open(args.save, "w") as f:
            f.write("<html><head></head><body>\n")
            f.write(fig_obs.to_html(full_html=False, include_plotlyjs="cdn"))
            f.write("<br>\n")
            f.write(fig_act.to_html(full_html=False, include_plotlyjs=False))
            f.write("</body></html>")
    else:
        # ブラウザで表示する場合は一時HTMLファイルを作成
        import tempfile
        import webbrowser

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            f.write("<html><head></head><body>\n")
            f.write(fig_obs.to_html(full_html=False, include_plotlyjs="cdn"))
            f.write("<br>\n")
            f.write(fig_act.to_html(full_html=False, include_plotlyjs=False))
            f.write("</body></html>")
            temp_path = f.name

        webbrowser.open(f"file://{temp_path}")


if __name__ == "__main__":
    main()
