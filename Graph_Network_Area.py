#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
area_combined.py

Интегрирует обход папок и чтение pickle-графов (аналогично GCN_nine_models),
добавляет глобальные фичи из AbcStats.json и обучает GNN для регрессии
по цели log1p(area) с нормализацией таргета и HuberLoss.

Скрипт делает:
  1) Преобразование networkx-графа -> PyTorch Geometric (one-hot тип узла + норм. степень).
  2) Сбор датасета из папок, фильтрацию по area>0, расчёт mu/sigma для нормализации log-таргета.
  3) GNN-модель на базе GraphSAGE + глобальные пуллинги + добавление глобальных фич.
  4) Обучение/валидация, ранняя остановка, сохранение лучшей модели.
  5) Финальные метрики на валидации и аналитические графики.
"""

from typing import Any, Dict, List, Tuple

import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    BatchNorm,
    SAGEConv,
    global_max_pool,
    global_mean_pool,
)

# -----------------------------
# 1. Graph-to-Data converter (reuse from GCN_nine_models)
# -----------------------------
def graph_from_nx(
    G: Any,
    type_to_idx: Dict[str, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Собирает (edge_index, X) для графа.

    Признаки узла: one-hot по типу + нормализованная степень вершины.

    Parameters
    ----------
    G : networkx.Graph
        Исходный граф (узлы содержат атрибут 'type').
    type_to_idx : dict[str, int]
        Отображение 'тип узла' -> индекс для one-hot.

    Returns
    -------
    edge_index : torch.LongTensor [2, E]
        Индексы рёбер в формате PyG.
    x : torch.FloatTensor [N, num_types + 1]
        Матрица признаков узлов (one-hot + норм. степень).
    """
    # Перенумеруем узлы 0..N-1
    mapping = {n: i for i, n in enumerate(G.nodes())}

    # Рёбра
    edges = [[mapping[u], mapping[v]] for u, v in G.edges()]
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )

    # One-hot по типам
    types = [G.nodes[n].get("type", "") for n in G.nodes()]
    idxs = [type_to_idx.get(t, 0) for t in types]
    onehot = F.one_hot(
        torch.tensor(idxs, dtype=torch.long),
        num_classes=len(type_to_idx),
    ).float()

    # Нормализованная степень вершины
    deg = np.array([d for _, d in G.degree()], dtype=np.float32)
    mu, sigma = deg.mean(), deg.std() + 1e-6
    deg_norm = (deg - mu) / sigma
    deg_tensor = torch.tensor(deg_norm, dtype=torch.float32).unsqueeze(1)

    # Итоговые признаки
    x = torch.cat([onehot, deg_tensor], dim=1)
    return edge_index, x


# -----------------------------
# 2. Dataset for area prediction
# -----------------------------
class GraphAreaDataset(Dataset):
    """
    Датасет для задачи регрессии площади (area).

    mapping: dict[str -> str]
        fid -> folder, где fid = os.path.join(folder, subdir)
    type_to_idx: dict[str -> int]
        Словарь для one-hot кодирования типов узлов.
    """

    def __init__(self, mapping: Dict[str, str], type_to_idx: Dict[str, int]) -> None:
        self.mapping = mapping
        self.file_ids = list(mapping.keys())
        self.type_to_idx = type_to_idx

    def __len__(self) -> int:
        return len(self.file_ids)

    def __getitem__(self, i: int) -> Data:
        fid = self.file_ids[i]
        fol = self.mapping[fid]
        nm = os.path.basename(fid)

        # 1) Граф
        pkl_path = os.path.join(fol, nm, f"{nm}.pkl")
        with open(pkl_path, "rb") as pf:
            G = pickle.load(pf)

        edge_index, x = graph_from_nx(G, self.type_to_idx)

        # 2) Статы из JSON
        js_path = os.path.join(fol, nm, f"{nm}AbcStats.json")
        with open(js_path, "r") as jf:
            stats = json.load(jf).get("abcStatsBalance", {})

        area = float(stats.get("area", 0.0))

        # 3) Глобальные фичи (простые структурные и из JSON)
        g_feats = torch.tensor(
            [
                G.number_of_nodes(),
                G.number_of_edges(),
                stats.get("inputs", 0),
                stats.get("outputs", 0),
                stats.get("lev", 0),
                stats.get("nd", 0),
                stats.get("edge", 0),
            ],
            dtype=torch.float32,
        )

        # 4) Таргет: log1p(area)
        y = torch.log1p(torch.tensor(area, dtype=torch.float32))

        return Data(x=x, edge_index=edge_index, y=y, g_feats=g_feats)


# -----------------------------
# 3. Prepare data loaders + compute mu_y, sigma_y
# -----------------------------
def prepare_data_area(
    max_files_per_folder: int = 200,
    batch_size: int = 4,
    num_workers: int = 12,
) -> Tuple[DataLoader, DataLoader, int, int, float, float]:
    """
    Формирует train/val подвыборки, считает статистики по log-таргету и
    создаёт DataLoader'ы.

    Returns
    -------
    train_loader, val_loader : DataLoader
        Лоадеры для обучения/валидации.
    num_types : int
        Количество уникальных типов узлов (для размера one-hot).
    num_valid : int
        Число валидных графов (area > 0).
    mu_y, sigma_y : float
        Среднее и std для нормализации log-таргета.
    """
    folders = [
        r"E:/0_no_firrtl_bench/1 (100-139in, 100-134 out)",
        r"E:/0_no_firrtl_bench/2_0 (50-69 in, 50-100 out)",
        r"E:/0_no_firrtl_bench/2_1 (70-89 in, 50-100 out)",
        r"E:/0_no_firrtl_bench/2_2 (90_100 in, 50-100 out)",
        r"E:/0_no_firrtl_bench/3_0 (50-69 in, 50-100 out)",
        r"E:/0_no_firrtl_bench/3_1 (70-89 in, 50-100 out)",
        r"E:/0_no_firrtl_bench/3_2 (90_100 in, 50-100 out)",
        r"E:/0_no_firrtl_bench/4_0 (50-69 in, 50-100 out)",
    ]

    # Собрать все fid -> folder
    mapping: Dict[str, str] = {}
    for fol in folders:
        if not os.path.isdir(fol):
            continue
        subs = sorted(
            d for d in os.listdir(fol) if os.path.isdir(os.path.join(fol, d))
        )[: max_files_per_folder]
        for sd in subs:
            mapping[os.path.join(fol, sd)] = fol

    # Фильтрация по area > 0 и сбор всех типов узлов
    types: set = set()
    valid: Dict[str, str] = {}

    for fid, fol in mapping.items():
        nm = os.path.basename(fid)
        js = os.path.join(fol, nm, f"{nm}AbcStats.json")
        if not os.path.isfile(js):
            continue

        with open(js, "r") as f:
            stats = json.load(f).get("abcStatsBalance", {})

        area = float(stats.get("area", 0.0))
        if area > 0.0:
            valid[fid] = fol
            with open(os.path.join(fol, nm, f"{nm}.pkl"), "rb") as pf:
                G = pickle.load(pf)
            for _, d in G.nodes(data=True):
                types.add(d.get("type", ""))

    type_to_idx = {t: i for i, t in enumerate(sorted(types))}
    ds = GraphAreaDataset(valid, type_to_idx)

    # Разбиение на train/val по индексам
    idxs = list(range(len(ds)))
    tr, te = train_test_split(idxs, test_size=0.2, random_state=42)
    train_ds, val_ds = Subset(ds, tr), Subset(ds, te)

    # Статистики log-таргета (на train)
    ys = torch.tensor(
        [train_ds[i].y.item() for i in range(len(train_ds))],
        dtype=torch.float32,
    )
    mu_y = ys.mean().item()
    sigma_y = ys.std().item() + 1e-6

    # Лоадеры
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=lambda b: Batch.from_data_list(b),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=lambda b: Batch.from_data_list(b),
    )

    return train_loader, val_loader, len(type_to_idx), len(valid), mu_y, sigma_y


# -----------------------------
# 4. GNN model for area
# -----------------------------
class AreaGNN(nn.Module):
    """
    Три слоя GraphSAGE + BatchNorm + Dropout.
    Агрегация графа: global mean/max pool.
    Дополнение: конкатенация глобальных фич (из JSON/структуры графа).
    Голова: MLP -> скаляр (лог-таргет).
    """

    def __init__(
        self,
        input_dim: int,
        g_feat_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.01,
    ) -> None:
        super().__init__()
        self.g_feat_dim = g_feat_dim

        self.convs = nn.ModuleList(
            [
                SAGEConv(input_dim, hidden_dim),
                SAGEConv(hidden_dim, hidden_dim),
                SAGEConv(hidden_dim, hidden_dim),
            ]
        )
        self.bns = nn.ModuleList([BatchNorm(hidden_dim) for _ in range(3)])

        self.lin1 = nn.Linear(2 * hidden_dim + g_feat_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.lin3 = nn.Linear(hidden_dim // 2, 1)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        g_feats: torch.Tensor,
    ) -> torch.Tensor:
        # GNN-блок
        h = x
        for i, conv in enumerate(self.convs):
            h = F.relu(self.bns[i](conv(h, edge_index)))
            h = self.drop(h)

        # Пуллинги
        hg_mean = global_mean_pool(h, batch)
        hg_max = global_max_pool(h, batch)

        # Привести глобальные фичи к батчу (если пришёл плоский тензор)
        if g_feats.dim() == 1:
            ng = g_feats.numel() // self.g_feat_dim
            g_feats = g_feats.view(ng, self.g_feat_dim)

        # Конкатенация и голова
        m = torch.cat([hg_mean, hg_max, g_feats], dim=1)
        m = F.relu(self.lin1(m))
        m = self.drop(m)
        m = F.relu(self.lin2(m))
        m = self.drop(m)
        return self.lin3(m).view(-1)


# -----------------------------
# 5. Training + analytics + plots
# -----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Данные + нормализация
    train_loader, val_loader, num_types, num_graphs, mu_y, sigma_y = prepare_data_area(
        max_files_per_folder=200,
        batch_size=4,
        num_workers=12,
    )
    train_size = int(0.8 * num_graphs)
    val_size = num_graphs - train_size

    # 2) Параметры модели
    input_dim = num_types + 1  # one-hot по типам + норм. степень
    g_feat_dim = 7

    model = AreaGNN(input_dim, g_feat_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    loss_fn = nn.HuberLoss()

    train_losses: List[float] = []
    val_losses: List[float] = []

    best_val = float("inf")
    no_imp = 0

    # --- ТРЕНИРОВКА ---
    for epoch in range(1, 41):
        # TRAIN
        model.train()
        tr_sum = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            y_norm = (batch.y - mu_y) / sigma_y
            pred_n = model(batch.x, batch.edge_index, batch.batch, batch.g_feats)
            loss = loss_fn(pred_n, y_norm)

            loss.backward()
            optimizer.step()

            tr_sum += loss.item() * batch.num_graphs

        tr_loss = tr_sum / train_size
        train_losses.append(tr_loss)

        # VAL
        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                y_norm = (batch.y - mu_y) / sigma_y
                pred_n = model(batch.x, batch.edge_index, batch.batch, batch.g_feats)
                val_sum += loss_fn(pred_n, y_norm).item() * batch.num_graphs

        val_loss = val_sum / val_size
        val_losses.append(val_loss)

        # LR scheduler + early stopping
        scheduler.step(val_loss)
        print(f"Epoch {epoch:02d}: Train={tr_loss:.4f}, Val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            no_imp = 0
            torch.save(model.state_dict(), "best_area_gnn.pth")
        else:
            no_imp += 1
            if no_imp >= 15:
                print(f"Early stopping at epoch {epoch}")
                break

    # --- ФИНАЛЬНАЯ ОЦЕНКА ---
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    model.load_state_dict(torch.load("best_area_gnn.pth", map_location=device))
    ys, ps = [], []

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred_n = model(batch.x, batch.edge_index, batch.batch, batch.g_feats)

            # Демасштабирование + expm1 (возврат из log1p)
            y_pred = torch.expm1(pred_n * sigma_y + mu_y).cpu().numpy()
            y_true = torch.expm1(batch.y).cpu().numpy()

            ps.append(y_pred)
            ys.append(y_true)

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"Final MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")

    # --- ПЛОТЫ ---
    os.makedirs("plots", exist_ok=True)

    # 1) Learning curves
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (norm-log)")
    plt.title("Learning Curves")
    plt.legend()
    plt.savefig("plots/learning_curves.png")
    plt.close()

    # 2) True vs Predicted Area
    plt.figure()
    plt.scatter(y_true, y_pred, s=10, alpha=0.6)
    mn, mx = y_true.min(), y_true.max()
    plt.plot([mn, mx], [mn, mx], "k--")
    plt.xlabel("True Area")
    plt.ylabel("Predicted Area")
    plt.title("True vs Predicted Area")
    plt.savefig("plots/true_vs_pred_area.png")
    plt.close()

    # 3) Residuals hist
    resid = y_pred - y_true
    plt.figure()
    plt.hist(resid, bins=50, edgecolor="black")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residuals Distribution")
    plt.savefig("plots/residuals_hist.png")
    plt.close()

    # 4) Residuals vs True
    plt.figure()
    plt.scatter(y_true, resid, s=10, alpha=0.6)
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("True Area")
    plt.ylabel("Residual")
    plt.title("Residuals vs True Area")
    plt.savefig("plots/residuals_vs_true.png")
    plt.close()

    # 5) Target distribution
    plt.figure()
    plt.hist(y_true, bins=30, edgecolor="black")
    plt.xlabel("True Area")
    plt.ylabel("Count")
    plt.title("Target Distribution")
    plt.savefig("plots/target_dist_area.png")
    plt.close()

    print("Построено и сохранено в ./plots/")
