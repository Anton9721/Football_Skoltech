"""
MatrixVisualizer — интерактивный heatmap для произвольной матрицы чисел.

Возможности:
  - Покраска через QuantileTransformer (по строкам / столбцам / глобально)
  - Палитра coolwarm + colorbar
  - Сортировка строк по кластерам HDBSCAN
  - Зум: подписи Y-оси становятся читаемы при увеличении
  - Выделение строк (Tap / BoxSelect) → список в текстовом поле + кнопка «Копировать»
  - Все виджеты — только Python-коллбэки (кроме JS-копирования в буфер)
  - Ячейки с raw==1.0 красятся в жёлтый, raw==0.0 — в сиреневый (поверх quantile)
  - Запуск: viz.show()  — поднимает локальный Bokeh-сервер в ноутбуке
"""

import threading
import webbrowser

import hdbscan
import numpy as np
import pandas as pd
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    ColorBar,
    ColumnDataSource,
    CustomJS,
    Div,
    LinearColorMapper,
    Select,
    TextAreaInput,
)
from bokeh.palettes import RdBu11
from bokeh.plotting import figure
from bokeh.server.server import BaseServer
from bokeh.server.tornado import BokehTornado
from bokeh.server.util import bind_sockets
from sklearn.preprocessing import QuantileTransformer as _SKLearnQT
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop


def quantile_transform_1d(arr: np.ndarray, n_quantiles: int = None) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n <= 1:
        return np.zeros(n)
    nq = min(n if n_quantiles is None else n_quantiles, n)
    qt = _SKLearnQT(n_quantiles=nq, output_distribution="uniform", subsample=n)
    return qt.fit_transform(arr.reshape(-1, 1)).ravel()


COLOR_PURE_ONE = "#ffff00"
COLOR_PURE_ZERO = "#cc66ff"


class MatrixVisualizer:
    """
    Параметры
    ----------
    matrix      : np.ndarray (N x K) или pd.DataFrame
    y_labels    : list[str] — подписи строк (признаки); None -> "feature_i"
    x_labels    : list[str] — подписи столбцов;         None -> "col_j"
    title       : str
    pure_one    : float | None  — raw-значение, которое красить жёлтым  (default 1.0)
    pure_zero   : float | None  — raw-значение, которое красить сиреневым (default 0.0)
    """

    def __init__(
        self,
        matrix,
        y_labels=None,
        x_labels=None,
        title: str = "Matrix Visualizer",
        pure_one: float = 1.0,
        pure_zero: float = 0.0,
    ):
        if isinstance(matrix, pd.DataFrame):
            self.matrix = matrix.values.astype(float)
            self.y_labels = list(matrix.index) if y_labels is None else list(y_labels)
            self.x_labels = list(matrix.columns) if x_labels is None else list(x_labels)
        else:
            self.matrix = np.array(matrix, dtype=float)
            N, K = self.matrix.shape
            self.y_labels = (
                [f"feature_{i}" for i in range(N)]
                if y_labels is None
                else list(y_labels)
            )
            self.x_labels = (
                [f"col_{j}" for j in range(K)] if x_labels is None else list(x_labels)
            )

        self.title = title
        self.pure_one = pure_one
        self.pure_zero = pure_zero
        self._ioloop = None
        self._hdbscan_sort()

    def _hdbscan_sort(self):
        N = self.matrix.shape[0]
        if N < 5:
            return
        embeddings = np.vstack(
            [quantile_transform_1d(self.matrix[i]) for i in range(N)]
        )
        try:
            min_cls = max(2, N // 15)
            labels = hdbscan.HDBSCAN(min_cluster_size=min_cls).fit_predict(embeddings)
            sort_key = np.where(labels == -1, int(labels.max()) + 1, labels)
            order = np.argsort(sort_key, kind="stable")
            self.matrix = self.matrix[order]
            self.y_labels = [self.y_labels[i] for i in order]
        except Exception as exc:
            print(f"[MatrixVisualizer] HDBSCAN пропущен: {exc}")

    def _color_matrix(self, axis: str = "rows") -> np.ndarray:
        N, K = self.matrix.shape
        cm = np.zeros((N, K))
        if axis == "rows":
            for i in range(N):
                cm[i] = quantile_transform_1d(self.matrix[i])
        elif axis == "columns":
            for j in range(K):
                cm[:, j] = quantile_transform_1d(self.matrix[:, j])
        else:
            cm = quantile_transform_1d(self.matrix.ravel()).reshape(N, K)
        return cm

    def show(self, port: int = 0, open_browser: bool = True):
        self.stop()

        sockets, actual_port = bind_sockets("localhost", port)

        app = Application(FunctionHandler(self._build_app))
        io_loop = IOLoop()
        tornado_app = BokehTornado({"/": app}, extra_websocket_origins=["*"])

        http_server = HTTPServer(tornado_app)
        http_server.add_sockets(sockets)

        server = BaseServer(io_loop, tornado_app, http_server)
        server.start()

        self._ioloop = io_loop

        url = f"http://localhost:{actual_port}"
        if open_browser:
            io_loop.add_callback(lambda: webbrowser.open(url))

        print(f"[MatrixVisualizer] {url}")

        threading.Thread(target=io_loop.start, daemon=True).start()

    def stop(self):
        if self._ioloop is not None:
            try:
                self._ioloop.add_callback(self._ioloop.stop)
            except Exception:
                pass
            self._ioloop = None

    def interpolate_palette(self, palette, n=256):
        def hex_to_rgb(h):
            h = h.lstrip('#')
            return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

        colors = [hex_to_rgb(c) for c in palette]
        xs = np.linspace(0, 1, len(colors))
        xi = np.linspace(0, 1, n)
        result = []
        for channel in range(3):
            result.append(np.interp(xi, xs, [c[channel] for c in colors]))
        return [f"#{int(r):02x}{int(g):02x}{int(b):02x}" for r, g, b in zip(*result)]

    def _build_app(self, doc):
        N, K = self.matrix.shape

        palette = self.interpolate_palette(list(reversed(RdBu11)), n=256)
        mapper = LinearColorMapper(palette=palette, low=0.0, high=1.0)

        mask_one = self.matrix == self.pure_one
        mask_zero = self.matrix == self.pure_zero

        def make_data(cm):
            xs, ys, cvs, rvs, feats, cols, clrs = [], [], [], [], [], [], []
            for i in range(N):
                for j in range(K):
                    xs.append(j)
                    ys.append(i)
                    cvs.append(float(cm[i, j]))
                    rvs.append(float(self.matrix[i, j]))
                    feats.append(self.y_labels[i])
                    cols.append(self.x_labels[j])
                    if mask_one[i, j]:
                        clrs.append(COLOR_PURE_ONE)
                    elif mask_zero[i, j]:
                        clrs.append(COLOR_PURE_ZERO)
                    else:
                        clrs.append(None)
            return dict(
                x=xs,
                y=ys,
                color_val=cvs,
                raw_val=rvs,
                feature=feats,
                col_name=cols,
                special_color=clrs,
            )

        source = ColumnDataSource(make_data(self._color_matrix("rows")))

        plot_w = 920
        plot_h = max(300, min(850, N * 13))

        p = figure(
            width=plot_w,
            height=plot_h,
            title=self.title,
            tools="tap,box_select,box_zoom,wheel_zoom,pan,reset",
            active_drag="box_zoom",
            active_scroll="wheel_zoom",
            x_range=(-0.5, K - 0.5),
            y_range=(-0.5, N - 0.5),
        )

        p.rect(
            x="x",
            y="y",
            width=1.0,
            height=1.0,
            source=source,
            fill_color={"field": "color_val", "transform": mapper},
            line_color=None,
            nonselection_fill_alpha=0.55,
            nonselection_fill_color={"field": "color_val", "transform": mapper},
        )

        def make_special_source(cm):
            xs, ys, clrs, rvs, feats, col_names = [], [], [], [], [], []
            for i in range(N):
                for j in range(K):
                    if mask_one[i, j] or mask_zero[i, j]:
                        xs.append(j)
                        ys.append(i)
                        clrs.append(
                            COLOR_PURE_ONE if mask_one[i, j] else COLOR_PURE_ZERO
                        )
                        rvs.append(float(self.matrix[i, j]))
                        feats.append(self.y_labels[i])
                        col_names.append(self.x_labels[j])
            return dict(
                x=xs, y=ys, color=clrs, raw_val=rvs, feature=feats, col_name=col_names
            )

        special_source = ColumnDataSource(
            make_special_source(self._color_matrix("rows"))
        )

        p.rect(
            x="x",
            y="y",
            width=1.0,
            height=1.0,
            source=special_source,
            fill_color="color",
            line_color=None,
        )

        p.xaxis.ticker = list(range(K))
        p.xaxis.major_label_overrides = {j: str(self.x_labels[j]) for j in range(K)}
        p.xaxis.major_label_orientation = 1.0
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None

        p.yaxis.ticker = list(range(N))
        p.yaxis.major_label_overrides = {i: str(self.y_labels[i]) for i in range(N)}
        fs = "7pt" if N > 50 else ("8pt" if N > 25 else "10pt")
        p.yaxis.major_label_text_font_size = fs

        p.add_layout(
            ColorBar(color_mapper=mapper, width=15, location=(0, 0), title=""), "right"
        )

        legend_html = Div(
            text=f"""
            <div style="display:flex; flex-direction:column;
                        justify-content:space-between;
                        height:{max(200, min(700, plot_h - 80))}px;
                        font-size:12px; padding:4px 0;">
                <div style="display:flex; align-items:center; gap:6px;">
                    <div style="width:16px;height:16px;
                                background:{COLOR_PURE_ONE};
                                border:1px solid #aaa;"></div>
                    Чистый 1
                </div>
                <div style="display:flex; align-items:center; gap:6px;">
                    <div style="width:16px;height:16px;
                                background:{COLOR_PURE_ZERO};
                                border:1px solid #aaa;"></div>
                    Чистый 0
                </div>
            </div>
        """,
            width=100,
        )

        axis_select = Select(
            title="Нормализация цвета:",
            value="rows",
            options=[
                ("rows", "По строкам (по умолчанию)"),
                ("columns", "По столбцам"),
                ("global", "Глобальная"),
            ],
            width=260,
        )
        selected_label = Div(text="<b>Выбранные признаки:</b>")
        text_area = TextAreaInput(value="", rows=4, width=plot_w - 180)
        copy_btn = Button(label="Копировать", button_type="success", width=160)

        copy_btn.js_on_click(
            CustomJS(
                args=dict(ta=text_area),
                code="""
            navigator.clipboard.writeText(ta.value).catch(() => {
                const el = document.createElement('textarea');
                el.value = ta.value; document.body.appendChild(el);
                el.select(); document.execCommand('copy');
                document.body.removeChild(el);
            });
        """,
            )
        )

        def on_axis_change(attr, old, new):
            cm_new = self._color_matrix(new)
            source.data["color_val"] = [
                float(cm_new[i, j]) for i in range(N) for j in range(K)
            ]

        axis_select.on_change("value", on_axis_change)

        def on_selection(attr, old, new):
            if not new:
                text_area.value = ""
                return
            seen: dict = {}
            for idx in new:
                seen[source.data["feature"][idx]] = None
            text_area.value = ", ".join(f"'{f}'" for f in seen)

        source.selected.on_change("indices", on_selection)

        doc.add_root(
            column(
                row(p, column(legend_html, axis_select)),
                column(selected_label, row(text_area, copy_btn)),
            )
        )
        doc.title = self.title


# viz = MatrixVisualizer(mat, y_labels=y_labels, x_labels=x_labels, title="Demo MatrixVisualizer")
# viz.show()
