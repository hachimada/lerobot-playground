## lerobot-playground

This repository uses the [`huggingface/lerobot`](https://github.com/huggingface/lerobot) project as a Git submodule for experimentation. Make sure to initialize the submodule after cloning:

```bash
git submodule update --init --recursive
```

Once the submodule is checked out you can work with the playground scripts (for example `python main.py`) while having direct access to the lerobot source under `lerobot/`.

### Updating the lerobot submodule

To pull the latest lerobot revisions referenced by this playground:

```bash
git submodule update --remote lerobot
```

If you want to update to a specific commit, check out that commit inside `lerobot/` and record it with `git add lerobot`.

### Installing lerobot into the current environment (with `uv`)

1. Create a project-local virtual environment managed by [`uv`](https://docs.astral.sh/uv/):

   ```bash
   uv venv
   source .venv/bin/activate
   ```

2. Install the lerobot submodule in editable mode so local changes under `lerobot/` are instantly available:

   ```bash
   uv pip install -e "./lerobot[all]"
   ```

3. (Optional) install the playground helpers too:

   ```bash
   uv pip install -e .
   ```

## setup motor

```bash
lerobot-setup-motors \
    --teleop.type=so101_leader \
    --teleop.port==/dev/ttyACM1

lerobot-setup-motors \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1
```

## calibrate

```bash
lerobot-calibrate \
--teleop.type=so101_leader \
--teleop.port=/dev/ttyACM0 \
--teleop.id=so101_leader_arm

lerobot-calibrate \
--robot.type=so101_follower \
--robot.port=/dev/ttyACM1 \
--robot.id=so101_follower_arm
```

## teleop

### using realsense

```bash
lerobot-teleoperate \
--robot.type=so101_follower \
--robot.port=/dev/tty.ACM1 \
--robot.id=so101_follower_arm \
--teleop.type=so101_leader \
--teleop.port=/dev/tty.ACM0 \
--teleop.id=so101_leader_arm \
--display_data=true \
--robot.cameras="{ front: {type: intelrealsense, serial_number_or_name: 829212070069, width: 640, height: 480, fps: 30}}" 
```

### using web-cam

```bash
lerobot-teleoperate \
--robot.type=so101_follower \
--robot.port=/dev/ttyACM1 \
--robot.id=so101_follower_arm \
--teleop.type=so101_leader \
--teleop.port=/dev/ttyACM0 \
--teleop.id=so101_leader_arm \
--display_data=true \
--robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}"
```

## record dataset

```bash
lerobot-record \
--robot.type=so101_follower \
--robot.port=/dev/ttyACM1 \
--robot.id=so101_follower_arm \
--teleop.type=so101_leader \
--teleop.port=/dev/ttyACM0 \
--teleop.id=so101_leader_arm \
--display_data=true \
--robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
--dataset.num_episodes=50 \
--dataset.single_task="pickup red object and place it to black box" \
--dataset.episode_time_s=15 \
--dataset.reset_time_s=10 \
--dataset.repo_id=${HF_USER}/so101_pick_place_red_1 \
--dataset.push_to_hub=false # default is true
--play_sounds=false # 音声ガイダンスの有無
--resume=true # 既存データセットへの追記モード
```


## train

```bash
lerobot-train \
--dataset.repo_id=${HF_USER}/so101_pick_place_red_1 \
--policy.type=act \
--output_dir=outputs/train/act_so101_test \
--job_name=act_so101_test \
--policy.device=cuda \
--wandb.enable=false \
--policy.repo_id=${HF_USER}/act_so101_pick_place_red_1
```

## inference

```bash
lerobot-record \
--robot.type=so101_follower \
--robot.port=/dev/ttyACM1 \
--robot.id=so101_follower_arm \
--robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
--display_data=true \
--dataset.repo_id=${HF_USER}/eval_record-test3 \
--dataset.single_task="pick red usb and place it to black box" \
--policy.path=${HF_USER}/my_policy
```

## visualize local dataset

```bash
uv run lerobot-dataset-viz \
--repo-id dummy  \
--root path/to/your/local/dataset \
--episode-index 4 \
--video-backend pyav
```

- --root: データセットのルートディレクトリへの完全なパスを指定します。（例: lerobot_dataset_15hz）
- --repo-id: データセットの識別子として使われます。ログ出力や、将来的にHubにプッシュする際などに使われますが、ローカルのパスを解決するためには使われません。

## update dataset to new Lerobot format

```bash
python src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
    --repo-id=lerobot/pusht \
    --root=/path/to/local/dataset/directory
    --push-to-hub=false
```

- スクリプトは、--rootで指定されたパスにデータセットが存在するかどうかを確認します。
- もしローカルにデータセットが存在しない場合、--repo-idで指定されたHugging Face Hubのリポジトリから、v2.1バージョンのデータセットをダウンロードします。