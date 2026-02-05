# GenesisRL_2025

Genesisシミュレータを用いた自律移動ロボット環境の構築および強化学習への適用

## Genesis環境構築とKachaka実装ガイド

Ubuntu 22.04.5 LTS において、物理シミュレータGenesisの環境を構築し、Kachakaロボットを実装する手順をまとめたものです。

## 目次

- [GenesisRL\_2025](#genesisrl_2025)
  - [Genesis環境構築とKachaka実装ガイド](#genesis環境構築とkachaka実装ガイド)
  - [目次](#目次)
  - [1. システム更新と基本ツールのインストール](#1-システム更新と基本ツールのインストール)
  - [2. NVIDIA CUDA ToolkitとPyTorchのインストール](#2-nvidia-cuda-toolkitとpytorchのインストール)
    - [NVIDIA CUDA Toolkitのインストール](#nvidia-cuda-toolkitのインストール)
    - [PyTorchのインストール](#pytorchのインストール)
  - [3. Genesisのインストール](#3-genesisのインストール)
  - [4. Genesisでの実行](#4-genesisでの実行)
  - [5. ROS2 Humbleのインストール](#5-ros2-humbleのインストール)
    - [はじめにUbuntuの更新](#はじめにubuntuの更新)
    - [必要なパッケージのインストールとレポジトリの追加](#必要なパッケージのインストールとレポジトリの追加)
    - [ROS 2 Humbleのインストール](#ros-2-humbleのインストール)
    - [環境設定とビルドツールのインストール](#環境設定とビルドツールのインストール)
  - [6. kachakaモデル(URDF)の作成](#6-kachakaモデルurdfの作成)
  - [7. トラブルシューティング](#7-トラブルシューティング)
    - [OpenGLエラー (Context Invalid等)](#openglエラー-context-invalid等)

---

## <div id="1-システム更新と基本ツールのインストール">1. システム更新と基本ツールのインストール

まずシステムを最新の状態にし、pythonなどの基本的なツールをインストールします。

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install build-essential python3-pip python3-venv git curl gnupg lsb-release
```

---

## <div id="2-NVIDIA-CUDA-ToolkitとPyTorchのインストール">2. NVIDIA CUDA ToolkitとPyTorchのインストール

### NVIDIA CUDA Toolkitのインストール

GenesisをGPUバックエンドで動作させるため、Ubuntu 22.04に対応したCUDAを公式サイトからインストールします。  
[PyTorch](https://pytorch.org/get-started/locally/)のサイトから対応しているCUDAのバージョンをインストールしてください。  
使っているGPUによって対応していないバージョンがあるのでNVIDIAの公式から確認してください。  
RTX2000番台以降のGPUを使用している場合は基本的に問題ないと思われます。  

[CUDAの旧バージョンのインストールページ](https://developer.nvidia.com/cuda-toolkit-archive)  
こちらのサイトからPyTorchが対応しているバージョンをインストールできます。

### PyTorchのインストール

[PyTorch](https://pytorch.org/get-started/locally/)のサイトから先程インストールしたCUDAのバージョンに合わせてインストールしてください

---

## <div id="3-Genesisのインストール">3. Genesisのインストール

pipでGenesisをインストールします。

```bash
pip install genesis-world
```

---

## <div id="4-Genesisでの実行">4. Genesisでの実行

GenesisのGithubからリポジトリをクローンする。

```bash
cd
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
```

動作確認として、Genesis/example/tutorialsの中のhello_genesis.pyを実行する。  
実行する際に、スクリプト内のgs.cpuをgs.gpuに変更するとGPUを使用します。  

```bash
cd Genesis
python3 example/tutorials/hello_genesis.py
```

ターミナルにFPSなどが表示されれば成功です。  
kachaka_RLフォルダやkachaka_lidar.pyをGenesis/example/に移動し、URDFファイルのパスの修正を行うことで実行できるようになります。

---

## <div id="5-ROS2-Humbleのインストール">5. ROS2 Humbleのインストール

kachakaのモデルデータ（URDF）を作成するためにROS2が必要なためインストールします。

### はじめにUbuntuの更新

```bash
sudo apt update && sudo apt upgrade -y
```

### 必要なパッケージのインストールとレポジトリの追加

```bash
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

### ROS 2 Humbleのインストール

```bash
sudo apt update
sudo apt install ros-humble-desktop
```

### 環境設定とビルドツールのインストール

```bash
source /opt/ros/humble/setup.bash
sudo apt install python3-colcon-common-extensions ros-humble-xacro
```

---

## <div id="6-kachakaモデル(URDF)の作成">6. kachakaモデル(URDF)の作成

Kachakaの公式GithubリポジトリからXacroファイルを取得し、Genesisで読み込めるURDF形式に変換します。
ただし、本研究で用いる機体は一部変更を加えているため、このレポジトリにあるファイルを用いてください。

```bash
cd
# ROS2環境をロード
source /opt/ros/humble/setup.bash

# リポジトリ取得
git clone https://github.com/pf-robotics/kachaka-api.git
cd kachaka-api/ros2/

# ビルド（kachaka_descriptionパッケージのみ）
colcon build --packages-select kachaka_description
source install/local_setup.bash

xacro kachaka_description/robot/kachaka.urdf.xacro > kachaka.urdf
```

turtlebot3もレポジトリ及びxacroファイルなどが公開されているため、同様手順で変換できます。

---

## <div id="7-トラブルシューティング">7. トラブルシューティング

### OpenGLエラー (Context Invalid等)

描画バックエンドの設定が必要です。スクリプト実行前に以下をエクスポートしてください。

```bash
export MUJOCO_GL=glx
export PYOPENGL_PLATFORM=glx
```

---
