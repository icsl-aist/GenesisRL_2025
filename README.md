# GenesisRL_2025
Genesisシミュレータを用いた自律移動ロボット環境の構築および強化学習への適用

## Genesis環境構築とKachaka実装ガイド
Ubuntu 22.04.5 LTS において、物理シミュレータGenesisの環境を構築し、Kachakaロボットを実装する手順をまとめたものです。

## 目次
- [1. システム更新と基本ツールのインストール](#1-システム更新と基本ツールのインストール)
- [2. NVIDIA CUDA ToolkitとPyTorchのインストール](#2-NVIDIA-CUDA-ToolkitとPyTorchのインストール)
- [3. Genesisのインストール](#3-Genesisのインストール)
- [4. Genesisでの実行](#4-Genesisでの実行)
- [5. ROS2 Humbleのインストール](#5-ROS2-Humbleのインストール)
- [6. Kachakaモデル(URDF)の作成](#6-Kachakaモデル(URDF)の作成)
- [7. トラブルシューティング](#7-トラブルシューティング)


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

## <div id="5-ROS2-Humbleのインストール">5. ROS2 Humbleのインストール


---
## <div id="6-Kachakaモデル(URDF)の作成">6. Kachakaモデル(URDF)の作成

---
## <div id="7-トラブルシューティング">7. トラブルシューティング

---
