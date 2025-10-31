#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Графовая нейросеть для предсказания задержки комбинационной схемы.

Скрипт выполняет:
1) Преобразование networkx-графа в форматы PyTorch Geometric (one-hot тип узла + норм. степень).
2) Формирование датасета из .pkl-графов, фильтрацию нулевых таргетов, нормализацию цели.
3) Обучение модели GraphSAGE с global mean/max pool и полноценной «головой» с BatchNorm.
4) Валидацию, сохранение лучшей модели, финальные метрики и аналитические графики.

Структура кода сохранена как в исходнике; изменены только оформление и читаемость.
"""

from typing import Any, Dict, List, Tuple

import json
import multiprocessing as mp
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import (
    BatchNorm,
    SAGEConv,
    global_max_pool,
    global_mean_pool,
)


# -----------------------------
# 1. Преобразование графа (networkx -> PyG Data-поля)
# -----------------------------
def graph_from_nx(
    G: Any,
    type_to_idx: Dict[str, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Собирает edge_index и матрицу признаков X для графа.

    Признаки узлов: one-hot по типу + нормализованная степень вершины.

    Parameters
    ----------
    G : networkx.Graph
        Исходный граф (узлы содержат атрибут 'type').
    type_to_idx : dict[str, int]
        Словарь "тип узла -> индекс" для one-hot.

    Returns
    -------
    edge_index : torch.LongTensor (shape [2, E])
        Индексы рёбер в формате PyG.
    x : torch.FloatTensor (shape [N, num_types + 1])
        Признаки узлов (one-hot + норм. степень).
    """
    # Отображение исходных идентификаторов узлов в 0..N-1
    mapping = {n: i for i, n in enumerate(G.nodes())}

    # Рёбра (ориентированные/неориентированные — как в исходном графе)
    edges = [[mapping[src], mapping[dst]] for src, dst in G.edges()]
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )

    # One-hot по типу узла
    types = [G.nodes[n].get("type", "") for n in G.nodes()]
    idxs = [type_to_idx.get(t, 0) for t in types]
    onehot = F.one_hot(
        torch.tensor(idxs, dtype=torch.long),
        num_classes=len(type_to_idx),
    ).float()

    # Нормализованная степень вершины
    deg = np.array([d for _, d in G.degree()], dtype=np.float32)
    mu_deg = deg.mean()
    sigma_deg = deg.std() + 1e-6
    deg_norm = (deg - mu_deg) / sigma_deg
    deg_tensor = torch.tensor(deg_norm, dtype=torch.float32).unsqueeze(1)

    # Финальная матрица признаков
    x = torch.cat([onehot, deg_tensor], dim=1)
    return edge_index, x


# -----------------------------
# 2. Класс датасета
# -----------------------------
class GraphPickleDataset(Dataset):
    """
    Датасет, который читает графы из структуры папок и собирает PyG Data.
    """

    def __init__(self, mapping: Dict[str, str], type_to_idx: Dict[str, int]) -> None:
        self.mapping = mapping
        self.file_ids = list(mapping.keys())
        self.type_to_idx = type_to_idx

    def __len__(self) -> int:
        return len(self.file_ids)

    def __getitem__(self, i: int) -> Data:
        fid = self.file_ids[i]
        folder = self.mapping[fid]
        name = os.path.basename(fid)

        pkl_path = os.path.join(folder, name, f"{name}.pkl")

        # Читаем граф из pickle
        with open(pkl_path, "rb") as pf:
            G = pickle.load(pf)

        # Собираем признаки и рёбра
        edge_index, x = graph_from_nx(G, self.type_to_idx)

        # Таргет: задержка из graph атрибута или из JSON
        delay = G.graph.get("delay", None)
        if delay is None:
            js_path = os.path.join(folder, name, f"{name}AbcStats.json")
            if os.path.exists(js_path):
                with open(js_path, "r") as jf:
                    data = json.load(jf)
                delay = data.get("abcStatsBalance", {}).get("delay", 0.0)
            else:
                delay = 0.0

        return Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor(float(delay)),
        )


# -----------------------------
# 3. Collate-функция для DataLoader
# -----------------------------
def custom_collate(batch: List[Data]) -> Batch:
    """
    Склеивает список графов в Batch, пропуская None-элементы.
    """
    valid = [b for b in batch if b is not None]
    return Batch.from_data_list(valid) if valid else Batch()


# -----------------------------
# 4. Подготовка данных (фильтрация нулевых таргетов)
# -----------------------------
def prepare_data(
    max_files_per_folder: int = 200,
    batch_size: int = 4,
    num_workers: int = 16,
) -> Tuple[GeoDataLoader, GeoDataLoader, int, float, float]:
    """
    Собирает train/test выборки из набора папок, сохраняет метаданные,
    возвращает DataLoader'ы и статистики по таргету.

    Возвращает:
        train_loader, test_loader, num_types, mu_y, sigma_y
    """
    folders = [
        "E:/0_no_firrtl_bench/1 (100-139in, 100-134 out)",
        "E:/0_no_firrtl_bench/2_0 (50-69 in, 50-100 out)",
        "E:/0_no_firrtl_bench/2_1 (70-89 in, 50-100 out)",
        "E:/0_no_firrtl_bench/2_2 (90_100 in, 50-100 out)",
        "E:/0_no_firrtl_bench/3_0 (50-69 in, 50-100 out)",
        "E:/0_no_firrtl_bench/3_1 (70-89 in, 50-100 out)",
        "E:/0_no_firrtl_bench/3_2 (90_100 in, 50-100 out)",
        "E:/0_no_firrtl_bench/4_0 (50-69 in, 50-100 out)",
    ]

    # Собираем {полный_путь_к_экземпляру: папка}
    mapping: Dict[str, str] = {}
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        subdirs = sorted(
            d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))
        )
        subdirs = subdirs[: max_files_per_folder]
        for d in subdirs:
            pkl = os.path.join(folder, d, f"{d}.pkl")
            if os.path.isfile(pkl):
                mapping[os.path.join(folder, d)] = folder

    # Фильтруем сэмплы с delay > 0 и собираем множество типов узлов
    types: set = set()
    valid: Dict[str, str] = {}

    for fid, folder in mapping.items():
        name = os.path.basename(fid)
        pkl_path = os.path.join(folder, name, f"{name}.pkl")

        with open(pkl_path, "rb") as pf:
            G = pickle.load(pf)

        delay = G.graph.get("delay", None)
        if delay is None:
            js_path = os.path.join(folder, name, f"{name}AbcStats.json")
            if os.path.exists(js_path):
                with open(js_path, "r") as jf:
                    data = json.load(jf)
                delay = data.get("abcStatsBalance", {}).get("delay", 0.0)
            else:
                delay = 0.0

        if float(delay) > 0.0:
            valid[fid] = folder
            for _, node_data in G.nodes(data=True):
                types.add(node_data.get("type", ""))

    type_to_idx = {t: i for i, t in enumerate(sorted(types))}
    dataset = GraphPickleDataset(valid, type_to_idx)

    # Трейн/тест разбиение по индексам
    idxs = list(range(len(dataset)))
    tr, te = train_test_split(idxs, test_size=0.2, random_state=42)
    train_ds, test_ds = Subset(dataset, tr), Subset(dataset, te)

    # Статы по таргету (на трене)
    delays = np.array([d.y.item() for d in train_ds], dtype=np.float32)
    mu_y = float(delays.mean())
    sigma_y = float(delays.std() + 1e-6)

    # Сохраняем метаданные для инференса
    with open("metadata.json", "w") as mf:
        json.dump(
            {
                "type_to_idx": type_to_idx,
                "mu_y": mu_y,
                "sigma_y": sigma_y,
            },
            mf,
        )

    # Лоадеры
    train_loader = GeoDataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=custom_collate,
    )
    test_loader = GeoDataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=custom_collate,
    )

    return train_loader, test_loader, len(types), mu_y, sigma_y


# -----------------------------
# 5. Модель: GraphSAGE + голова с BatchNorm (регрессия)
# -----------------------------
class FinalGNN(nn.Module):
    """
    Три слоя GraphSAGE, рез. соединения, затем глобальные пуллинги (mean+max)
    и MLP-голова под регрессию задержки.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.01):
        super().__init__()

        # Блоки свёрток по графу
        self.layers = nn.ModuleList(
            [
                SAGEConv(input_dim, hidden_dim),
                SAGEConv(hidden_dim, hidden_dim),
                SAGEConv(hidden_dim, hidden_dim),
            ]
        )
        self.bns = nn.ModuleList([BatchNorm(hidden_dim) for _ in range(3)])

        # «Голова»
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.bn_head = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        # GNN-блок c остаточными соединениями
        h = F.relu(self.bns[0](self.layers[0](x, edge_index)))
        h = self.drop(h)

        for i in range(1, 3):
            h_new = F.relu(self.bns[i](self.layers[i](h, edge_index)))
            h = self.drop(h + h_new)

        # Глобальные пуллинги + MLP-голова
        pooled = torch.cat(
            [global_mean_pool(h, batch), global_max_pool(h, batch)],
            dim=1,
        )
        out = F.relu(self.fc1(pooled))
        out = self.bn_head(out)
        out = self.drop(out)

        out = F.relu(self.fc2(out))
        out = self.drop(out)

        return self.fc3(out)


# -----------------------------
# 6. Обучение + метрики + графики
# -----------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Данные
    tl, vl, num_types, mu_y, sigma_y = prepare_data(
        max_files_per_folder=200,
        batch_size=4,
        num_workers=16,
    )

    # Модель/оптимизатор/шедулер
    model = FinalGNN(num_types + 1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )
    total_steps = 50 * len(tl)  # используется только как справочная переменная
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35)
    criterion = nn.SmoothL1Loss()

    train_losses: List[float] = []
    val_losses: List[float] = []

    best = float("inf")
    no_imp = 0

    # --- Тренировка ---
    for epoch in range(1, 41):
        model.train()
        t_loss = 0.0

        for batch in tl:
            batch = batch.to(device)

            y = batch.y.view(-1)
            y_norm = (y - mu_y) / sigma_y

            pred = model(batch.x, batch.edge_index, batch.batch).view(-1)
            loss = criterion(pred, y_norm)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            t_loss += loss.item()

        t_loss /= len(tl)
        train_losses.append(t_loss)

        # --- Валидация ---
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in vl:
                batch = batch.to(device)

                y = batch.y.view(-1)
                y_norm = (y - mu_y) / sigma_y

                pred = model(batch.x, batch.edge_index, batch.batch).view(-1)
                v_loss += criterion(pred, y_norm).item()

        v_loss /= len(vl)
        val_losses.append(v_loss)

        print(f"Epoch {epoch:02d}: Train={t_loss:.4f}, Val={v_loss:.4f}")

        # Сохраняем лучшую по валидации модель
        if v_loss < best:
            best = v_loss
            no_imp = 0
            torch.save(model.state_dict(), "final_gnn_model.pth")
        else:
            no_imp += 1
            if no_imp >= 10:
                print("Early stopping")
                break

    print(f"Best val = {best:.6f}")

    # --- Финальные метрики на валидации ---
    model.load_state_dict(torch.load("final_gnn_model.pth", map_location=device))

    yt: List[np.ndarray] = []
    yp: List[np.ndarray] = []

    with torch.no_grad():
        for batch in vl:
            batch = batch.to(device)
            pred_norm = model(batch.x, batch.edge_index, batch.batch).view(-1)
            pred = (pred_norm * sigma_y + mu_y).cpu().numpy()

            yp.append(pred)
            yt.append(batch.y.view(-1).cpu().numpy())

    yt_arr = np.concatenate(yt)
    yp_arr = np.concatenate(yp)

    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        mean_absolute_percentage_error,  # добавлен недостающий импорт
    )

    mae = mean_absolute_error(yt_arr, yp_arr)
    rmse = np.sqrt(mean_squared_error(yt_arr, yp_arr))
    r2 = r2_score(yt_arr, yp_arr)
    mape = mean_absolute_percentage_error(yt_arr, yp_arr) * 100.0

    print(
        f"Final MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}, MAPE={mape:.2f}%"
    )

    # --- Графики ---
    os.makedirs("plots", exist_ok=True)

    # 1) Learning curves
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Learning Curves")
    plt.savefig("plots/learning_curves.png")
    plt.close()

    # 2) True vs Pred
    plt.figure()
    plt.scatter(yt_arr, yp_arr, s=10, alpha=0.6)
    mn, mx = yt_arr.min(), yt_arr.max()
    plt.plot([mn, mx], [mn, mx], "k--")
    plt.xlabel("True Delay")
    plt.ylabel("Predicted Delay")
    plt.title("True vs Predicted")
    plt.savefig("plots/true_vs_pred.png")
    plt.close()

    # 3) Residuals hist
    res = yp_arr - yt_arr
    plt.figure()
    plt.hist(res, bins=50, edgecolor="black")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residuals Distribution")
    plt.savefig("plots/residuals_hist.png")
    plt.close()

    # 4) Residuals vs True
    plt.figure()
    plt.scatter(yt_arr, res, s=10, alpha=0.6)
    plt.axhline(0, color="k", linestyle="--")
    plt.xlabel("True Delay")
    plt.ylabel("Residual")
    plt.title("Residuals vs True")
    plt.savefig("plots/residuals_vs_true.png")
    plt.close()

    # 5) Target distribution
    plt.figure()
    plt.hist(yt_arr, bins=30, edgecolor="black")
    plt.xlabel("True Delay")
    plt.ylabel("Count")
    plt.title("Target Distribution")
    plt.savefig("plots/target_dist.png")
    plt.close()

    print("Plots saved in ./plots/")
