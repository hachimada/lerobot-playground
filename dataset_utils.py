#!/usr/bin/env python
"""

データセットから生のデータ（動画フレーム、関節位置）を抽出して、
pi05_simple_inference.pyのPI05Inferencerクラスをテストします。
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging

from pathlib import Path
import json
import pandas as pd
import numpy as np
import logging
import cv2
import torch


class LerobotDatasetHelper:
    """
    LeRobotデータセットを扱うためのユーティリティクラス
    
    Attributes:
        dataset_dir (Path): LeRobotデータセットのルートディレクトリ。
            dataset_dir
            ├── data/
            │   ├── chunk-000/
            │   │   ├── file-000.parquet
            │   │   ├── file-001.parquet
            │   │   └── ...
            │   ├── chunk-001/
            │   │   ├── file-000.parquet
            │   │   └── ...
            │   └── ...
            ├── meta/
            │   └── stats.json
            │   └── info.json
            └── ...
    """

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.parquet_dir = self.dataset_dir / "data"
        self.info_json = self.dataset_dir / "meta" / "info.json"
        self._episode_data: pd.DataFrame | None = None  # キャッシュ用
        self._cached_episode_index: int | None = (
            None  # キャッシュされたエピソードのインデックス
        )

        self.parquet_files: list[Path] = sorted(self.parquet_dir.rglob("*.parquet"))
        with open(self.info_json, "r") as f:
            info = json.load(f)
        self.total_episodes = info["total_episodes"]
        self.fps = info["fps"]
        self.keys: list[str] = list(info["features"].keys())

    def show_dataset_info(self):
        """
        データセットの基本情報を表示する
        """
        print(f"Total Parquet Files: {len(self.parquet_files)}")
        print(f"Total Episodes: {self.total_episodes}")
        print(f"FPS: {self.fps}")
        print(f"Available Keys: {self.keys}")

    def get_episode_info(
        self,
        episode_index: int,
    ) -> dict:
        """
        指定されたエピソードの基本情報を取得
        """
        episode_data = self.get_episode_data(
            episode_index=episode_index,
        )
        if episode_data.empty:
            return {}
        episode_info = {
            "episode_index": episode_index,
            "num_frames": episode_data.shape[0],
            "length_seconds": episode_data.shape[0] / self.fps,
        }
        return episode_info

    def show_episode_info(
        self,
        target_episode_index: int,
    ):
        # データを抽出
        episode_data = self.get_episode_data(
            episode_index=target_episode_index,
        )

        # 結果を表示
        if not episode_data.empty:
            print(f"エピソード {target_episode_index} のデータ:")
            info = self.get_episode_info(
                episode_index=target_episode_index,
            )
            print(
                f"  フレーム数: {info['num_frames']}, 約 {info['length_seconds']:.2f} 秒"
            )
            frame_range = self.get_dataset_frame_index_range(
                episode_index=target_episode_index,
            )
            print(
                f"  データセット内のフレームインデックス範囲: {frame_range[0]} - {frame_range[1]}"
            )
        else:
            print(f"エピソード {target_episode_index} のデータは見つかりませんでした。")

    def get_episode_data(
        self,
        episode_index: int,
    ) -> pd.DataFrame:
        """
        指定されたエピソードのデータを取得
        """
        episode_dfs = []
        # `read_parquet`のためのフィルタ条件を作成
        filters = [("episode_index", "==", episode_index)]

        for file_path in self.parquet_files:
            # filters引数を使って、ファイルから特定のepisode_indexを持つ行のみを読み込む
            df = pd.read_parquet(file_path, filters=filters)
            if not df.empty:
                episode_dfs.append(df)

        # データが見つからなかった場合
        if not episode_dfs:
            return pd.DataFrame()

        # 複数のDataFrameを一つに結合して返す
        combined_df = pd.concat(episode_dfs, ignore_index=True)
        # キャッシュとして保存
        self._episode_data = combined_df
        self._cached_episode_index = episode_index

        return combined_df

    def _get_col_data(
        self,
        col: str,
        episode_index: int,
    ) -> np.ndarray:
        # キャッシュが無い、空、または異なるエピソードの場合に再取得
        if (
            self._episode_data is None
            or self._episode_data.empty
            or self._cached_episode_index != episode_index
        ):
            self.get_episode_data(
                episode_index=episode_index,
            )
        if self._episode_data is None or self._episode_data.empty:
            raise ValueError(f"Episode {episode_index} not found in dataset")
        if col not in self._episode_data.columns:
            raise ValueError(f"Column {col} not found in dataset")
        col_data = np.stack(self._episode_data[col].to_numpy())
        return col_data
    
    def get_dataset_frame_index_range(
        self,
        episode_index: int,
    ) -> tuple[int, int]:
        """
        指定されたエピソードのデータセット内のフレームインデックス範囲を取得
        """
        episode_data = self.get_episode_data(
            episode_index=episode_index,
        )
        if episode_data.empty:
            raise ValueError(f"Episode {episode_index} not found in dataset")
        start_index = episode_data["index"].min()
        end_index = episode_data["index"].max()
        return start_index, end_index
        

    def get_action(
        self,
        action_col: str,
        episode_index: int,
    ) -> np.ndarray:
        return self._get_col_data(
            col=action_col,
            episode_index=episode_index,
        )

    def get_observation(
        self,
        observation_col: str,
        episode_index: int,
    ) -> np.ndarray:
        return self._get_col_data(
            col=observation_col,
            episode_index=episode_index,
        )



def extract_raw_data_from_dataset(
    dataset: LeRobotDataset, frame_index: int = 0, visualize: bool = True
) -> dict:
    """
    データセットから生のデータを抽出（LeRobotDatasetを使用）

    Args:
        dataset: LeRobotDataset
        frame_index: 抽出するフレームのインデックス
        visualize: 画像を可視化するか（デフォルト: True）

    Returns:
        生のデータを含む辞書
            - observations: 観測データの辞書（PI05Inferencer.predictに直接渡せる形式）
            - task: タスク文字列
            - raw_data_info: 抽出元の情報
    """
    # データセットからフレームを取得
    frame = dataset[frame_index]

    # タスク情報を抽出
    task = frame.get("task", "default task")
    if isinstance(task, bytes):
        task = task.decode("utf-8")
        
    # 観測データを抽出（生データ形式に変換）
    observations = {}

    for key, value in frame.items():
        if key.startswith("observation."):
            if isinstance(value, torch.Tensor):
                # 画像データの場合
                if "image" in key:
                    # (C, H, W) -> (H, W, C) に変換
                    value_hwc = value.permute(1, 2, 0)
                    # [0, 1] -> [0, 255] に変換
                    value_uint8 = (value_hwc * 255).clamp(0, 255).to(torch.uint8)

                    # numpy配列に変換
                    image_np = value_uint8.cpu().numpy()

                    # RGB -> BGR に変換（cv2.imshowと実際のカメラはBGRを出力するため）
                    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                    # PI05Inferencer.predictに渡せる形式で保存
                    # BGR (H, W, 3) uint8 形式のnumpy配列（実際のカメラから取得される形式）
                    observations[key] = image_bgr

                else:
                    # 状態データの場合
                    state_np = value.cpu().numpy()

                    observations[key] = state_np
            else:
                observations[key] = value

    raw_data = {
        "observations": observations,
        "task": task,
        "raw_data_info": {
            "dataset": dataset.repo_id,
            "frame_index": frame_index,
        },
    }

    return raw_data


import tqdm
def test_all_frames(
    dataset: LeRobotDataset,
):

    frame_count = len(dataset)
    logging.info(f"Total frames in dataset: {frame_count}")
    for frame_idx in tqdm.tqdm(range(frame_count)):
        
        try:
            raw_data = extract_raw_data_from_dataset(
                dataset,
                frame_index=frame_idx,
                visualize=False,
            )
            logging.info(f"Extracted raw data for frame {frame_idx} successfully.")
        except Exception as e:
            logging.error(f"Error extracting raw data for frame {frame_idx}: {repr(e)}")
            


def main():
    """メイン関数"""
    init_logging()
    # dataset_dir = Path("datasets_kiyose/converted_kiyose_tool2")
    dataset_dir = Path("datasets_IO/0115_1-25")
    
    save_dir = Path("episode_videos") / dataset_dir.name
    save_dir.mkdir(parents=True, exist_ok=True)

    # データセットを読み込む
    dataset = LeRobotDataset(repo_id=str(dataset_dir), root=str(dataset_dir), video_backend="pyav")
    helper = LerobotDatasetHelper(dataset_dir=dataset_dir)
    helper.show_dataset_info()

    for ep_idx in range(helper.total_episodes):

        # if ep_idx not in [244]:  # kiyose_tool2データセットの特定エピソードのみ処理
        #     continue
        if ep_idx not in [1]:
            continue
      
        helper.show_episode_info(target_episode_index=ep_idx)
        action = helper.get_action("action", ep_idx)
        logging.info(f"Episode {ep_idx} action shape: {action.shape}")

        observation = helper.get_observation("observation.state",ep_idx)
        logging.info(f"Episode {ep_idx} observation shape: {observation.shape}")

        ep_frame_start_index, ep_frame_end_index = helper.get_dataset_frame_index_range(ep_idx)

        logging.info(f"Episode {ep_idx} frame index range: {ep_frame_start_index} - {ep_frame_end_index} in dataset")

        ep_frames: dict[str, np.ndarray] = {}  # エピソード内のフレームデータを時系列で保持
        # tpdmでプログレスバーを表示
        for frame_idx in tqdm.tqdm(range(ep_frame_start_index, ep_frame_end_index + 1), desc=f"Processing Episode {ep_idx}"):

            raw_data = extract_raw_data_from_dataset(dataset, frame_idx)
            
            # フレームデータを収集
            observation_keys = raw_data["observations"].keys()
            for key in observation_keys:
                if "images" in key:
                    frame = raw_data["observations"][key]
                    if key not in ep_frames:
                        ep_frames[key] = np.empty((0, *frame.shape), dtype=frame.dtype)
                    ep_frames[key] = np.vstack((ep_frames[key], frame[np.newaxis, ...]))
        
        # フレームを結合してmp4として保存
        for key, frames in tqdm.tqdm(ep_frames.items(), desc=f"Saving videos for Episode {ep_idx}"):
            output_video_path = save_dir / Path(f"episode_{ep_idx}/{key.replace('.', '_')}.mp4")
            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                str(output_video_path),
                fourcc,
                helper.fps,
                (width, height),
            )
            for frame in frames:
                video_writer.write(frame)
            video_writer.release()

if __name__ == "__main__":
    main()