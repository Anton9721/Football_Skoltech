import io
import base64

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from sklearn.metrics import confusion_matrix

import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets


LABELS = ["team_left", "team_right", "goalkeeper"]
LABEL_COLORS = {
    "team_left":  "#3b82f6",
    "team_right": "#ef4444",
    "goalkeeper": "#22c55e",
}


# ─── utils ────────────────────────────────────────────────────────────────────

def _to_base64(path: str, max_side: int = 256) -> str:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max(w, h) / max_side
    if scale > 1:
        img = img.resize((int(w / scale), int(h / scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _to_str_labels(y) -> list[str]:
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.integer):
        return [LABELS[i] if 0 <= i < len(LABELS) else str(i) for i in y]
    return [str(l) for l in y]


def _sample(X, y, df, n, seed):
    X, y = np.asarray(X), np.asarray(y)
    df = df.reset_index(drop=True)
    if n is None or len(X) <= n:
        return X, y, df
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X), size=n, replace=False))
    return X[idx], y[idx], df.iloc[idx].reset_index(drop=True)


def reduce_embeddings(X: np.ndarray, method: str = "umap", seed: int = 42) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    if method == "tsne":
        return TSNE(
            n_components=2, perplexity=30,
            learning_rate="auto", init="pca", random_state=seed,
        ).fit_transform(X)
    if method == "umap":
        return umap.UMAP(
            n_components=2, n_neighbors=30,
            min_dist=0.1, metric="cosine", random_state=seed,
        ).fit_transform(X)
    raise ValueError(f"Unknown method: {method}")


# ─── embedding scatter ────────────────────────────────────────────────────────

def interactive_embedding_view(
    X,
    y,
    df: pd.DataFrame,
    method: str = "umap",
    sample_n: int = 3000,
    seed: int = 42,
    point_size: int = 6,
    preload_images: bool = True,
):
    """
    Интерактивный scatter plot эмбеддингов.
    Hover  — превью кропа (если preload_images=True).
    Клик   — увеличенное фото + метаданные в панели справа.

    Parameters
    ----------
    X               : (N, D) эмбеддинги
    y               : (N,)   метки int или str
    df              : DataFrame с колонками crop_path, game, frame_idx, player_id
    method          : 'umap' | 'tsne' | 'pca'
    sample_n        : кол-во точек (None = все)
    preload_images  : кодировать картинки заранее для hover
    """
    if "crop_path" not in df.columns:
        raise ValueError("df должен содержать колонку 'crop_path'")

    Xs, ys, dfs = _sample(X, y, df, sample_n, seed)
    labels_str = _to_str_labels(ys)

    print(f"Снижение размерности ({method.upper()}) для {len(Xs)} точек...")
    Z = reduce_embeddings(Xs, method=method, seed=seed)
    print("Готово.")

    meta_cols = [c for c in ["game", "frame_idx", "player_id", "split"] if c in dfs.columns]

    if preload_images:
        print("Кодирование изображений...")
        b64_list = []
        for path in dfs["crop_path"]:
            try:
                b64_list.append(_to_base64(path, max_side=96))
            except Exception:
                b64_list.append("")
        print("Готово.")
    else:
        b64_list = [""] * len(dfs)

    df_plot = pd.DataFrame({
        "x":         Z[:, 0],
        "y":         Z[:, 1],
        "label":     labels_str,
        "crop_path": dfs["crop_path"].values,
        "b64":       b64_list,
        **{c: dfs[c].astype(str).values for c in meta_cols},
    })

    # customdata порядок: [label, meta..., b64]
    custom_cols = ["label"] + meta_cols + ["b64"]
    customdata  = df_plot[custom_cols].values

    b64_idx = len(meta_cols) + 1

    hover_lines = ["<b>%{customdata[0]}</b>"]
    for i, col in enumerate(meta_cols, start=1):
        hover_lines.append(f"{col}: %{{customdata[{i}]}}")
    if preload_images:
        hover_lines.append(
            f'<br><img src="data:image/jpeg;base64,%{{customdata[{b64_idx}]}}" '
            f'style="width:88px;height:auto;border-radius:4px;margin-top:4px">'
        )
    hover_template = "<br>".join(hover_lines) + "<extra></extra>"

    # ── figure ───────────────────────────────────────────────────────────────
    fig = go.Figure()
    for lbl in LABELS:
        mask = df_plot["label"] == lbl
        if not mask.any():
            continue
        sub = df_plot[mask]
        fig.add_trace(go.Scatter(
            x=sub["x"], y=sub["y"],
            mode="markers",
            name=lbl,
            marker=dict(
                size=point_size,
                color=LABEL_COLORS.get(lbl, "#888"),
                opacity=0.75,
                line=dict(width=0),
            ),
            customdata=customdata[mask.values],
            hovertemplate=hover_template,
        ))

    fig.update_layout(
        title=f"{method.upper()} — {len(df_plot)} точек",
        height=680,
        plot_bgcolor="#f8f9fa",
        legend=dict(title="Класс", itemsizing="constant"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=50, b=20),
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="#e2e8f0"),
    )

    figw = go.FigureWidget(fig)

    # ── панель клика ──────────────────────────────────────────────────────────
    img_out  = widgets.Output(layout=widgets.Layout(width="250px"))
    meta_out = widgets.Output(layout=widgets.Layout(width="250px"))

    panel = widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:13px;font-weight:600;"
                "color:#334155;margin-bottom:6px'>←</div>"
            ),
            img_out,
            meta_out,
        ],
        layout=widgets.Layout(
            width="265px",
            min_height="300px",
            padding="12px",
            border="1px solid #e2e8f0",
        ),
    )

    def on_click(trace, points, _state):
        if not points.point_inds:
            return
        i   = points.point_inds[0]
        cd  = trace.customdata[i]
        lbl = cd[0]
        meta_vals = {meta_cols[j]: cd[j + 1] for j in range(len(meta_cols))}
        crop_path = df_plot.loc[
            (df_plot["x"] == trace.x[i]) & (df_plot["y"] == trace.y[i]),
            "crop_path",
        ].values
        crop_path = crop_path[0] if len(crop_path) else None

        with img_out:
            clear_output(wait=True)
            if crop_path:
                try:
                    b64 = _to_base64(crop_path, max_side=230)
                    color = LABEL_COLORS.get(lbl, "#888")
                    display(HTML(
                        f'<div style="border:3px solid {color};'
                        f'border-radius:6px;display:inline-block">'
                        f'<img src="data:image/jpeg;base64,{b64}"'
                        f' style="display:block;max-width:230px"></div>'
                    ))
                except Exception as e:
                    display(HTML(f'<span style="color:#ef4444">Ошибка: {e}</span>'))

        with meta_out:
            clear_output(wait=True)
            color = LABEL_COLORS.get(lbl, "#888")
            rows = "".join(
                f"<tr>"
                f"<td style='color:#64748b;padding:2px 6px;font-size:11px'>{k}</td>"
                f"<td style='padding:2px 6px;font-size:11px'>{v}</td>"
                f"</tr>"
                for k, v in meta_vals.items()
            )
            display(HTML(
                f"<div style='margin-top:8px'>"
                f"<span style='background:{color};color:white;padding:2px 10px;"
                f"border-radius:12px;font-size:12px'>{lbl}</span>"
                f"<table style='margin-top:6px;border-collapse:collapse'>{rows}</table>"
                f"</div>"
            ))

    for trace in figw.data:
        trace.on_click(on_click)

    display(widgets.HBox(
        [figw, panel],
        layout=widgets.Layout(align_items="flex-start"),
    ))
    return figw


# ─── confusion matrix ─────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, normalize: bool = False):
    """
    Интерактивная confusion matrix.

    Parameters
    ----------
    y_true, y_pred : int (0/1/2) или str метки
    normalize      : True — показывает доли, False — абсолютные числа
    """
    yt = _to_str_labels(y_true)
    yp = _to_str_labels(y_pred)

    present = [l for l in LABELS if l in set(yt) | set(yp)]
    cm = confusion_matrix(yt, yp, labels=present)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_show  = np.where(row_sums > 0, cm / row_sums, 0).round(3)
        title    = "Confusion Matrix (normalized)"
    else:
        cm_show = cm
        title   = "Confusion Matrix"

    df_cm = pd.DataFrame(cm_show, index=present, columns=present)

    fig = px.imshow(
        df_cm,
        text_auto=".2%" if normalize else "d",
        color_continuous_scale="Blues",
        title=title,
        labels=dict(x="Predicted", y="True"),
    )
    fig.update_layout(
        height=460,
        coloraxis_showscale=False,
        font=dict(size=13),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_traces(
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Value: %{z}<extra></extra>"
    )
    fig.show()
    return df_cm