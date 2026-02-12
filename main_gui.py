import sys
import time
import pandas as pd
import numpy as np
import io
import joblib
import os
import json
import base64
import html
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QFileDialog, QComboBox, QLabel, QSpinBox, 
    QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit, QTextBrowser,
    QMessageBox, QListWidget, QGroupBox, QProgressBar,
    QFormLayout, QDoubleSpinBox, QLineEdit, QStackedWidget, QAbstractButton,
    QScrollArea, QHeaderView, QAbstractItemView
)
from PyQt6.QtWidgets import QListWidgetItem
from PyQt6.QtCore import Qt, QRect, QThread, pyqtSignal, QEvent, QTimer
from PyQt6.QtGui import QPalette, QColor, QPainter

# Importando lógica existente
from data_handler import load_data, separate_features_target, identify_problem_type, split_data
from pipeline_builder import build_pipeline, SCALER_OPTIONS, SAMPLER_OPTIONS
from experiment_runner import evaluate_model, generate_plots, save_pipeline_to_buffer
from reporter import generate_report
from models.registry import get_model_specs_by_problem
from literature_content import get_literature_html


class ToggleSwitch(QAbstractButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(40, 20)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        radius = rect.height() / 2
        track_color = QColor(46, 125, 50) if self.isChecked() else QColor(200, 200, 200)
        knob_color = QColor(245, 245, 245)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(track_color)
        painter.drawRoundedRect(rect, radius, radius)

        knob_size = rect.height() - 4
        knob_x = rect.width() - knob_size - 2 if self.isChecked() else 2
        knob_rect = QRect(knob_x, 2, knob_size, knob_size)
        painter.setBrush(knob_color)
        painter.drawEllipse(knob_rect)


class LiteraturePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("literaturePanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._dragging = False
        self._drag_start_y = 0.0
        self._start_height = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.handle = QWidget(self)
        self.handle.setObjectName("literatureHandle")
        self.handle.setFixedHeight(32)
        self.handle.setCursor(Qt.CursorShape.SizeVerCursor)
        self.handle.installEventFilter(self)

        handle_layout = QHBoxLayout(self.handle)
        handle_layout.setContentsMargins(12, 0, 12, 0)
        handle_layout.addWidget(QLabel("Literatura"))
        handle_layout.addStretch()

        self.content = QWidget(self)
        self.content.setObjectName("literatureContent")
        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(12, 12, 12, 12)
        self.viewer = QTextBrowser(self)
        self.viewer.setObjectName("literatureViewer")
        self.viewer.setOpenExternalLinks(True)
        self.viewer.setHtml("<h3>Selecione um modelo para ver a literatura.</h3>")
        content_layout.addWidget(self.viewer)

        layout.addWidget(self.handle)
        layout.addWidget(self.content, 1)

        self.setFixedHeight(32)

    def eventFilter(self, obj, event):
        if obj is self.handle:
            if event.type() == QEvent.Type.MouseButtonPress:
                self._dragging = True
                self._drag_start_y = event.globalPosition().y()
                self._start_height = self.height()
                return True
            if event.type() == QEvent.Type.MouseMove and self._dragging:
                delta = event.globalPosition().y() - self._drag_start_y
                max_height = max(self.handle.height(), self.parent().height())
                new_height = int(self._start_height - delta)
                new_height = max(self.handle.height(), min(new_height, max_height))
                self.setFixedHeight(new_height)
                self.update_position()
                return True
            if event.type() == QEvent.Type.MouseButtonRelease:
                self._dragging = False
                return True
        return super().eventFilter(obj, event)

    def update_position(self):
        parent = self.parent()
        if parent is None:
            return
        rect = parent.rect()
        self.setGeometry(0, rect.height() - self.height(), rect.width(), self.height())

    def set_html(self, html_text):
        self.viewer.setHtml(html_text)

# --- Worker Thread para Processamento Pesado ---
class MLWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, df, target_column, problem_type, selected_models, params):
        super().__init__()
        self.df = df
        self.target_column = target_column
        self.problem_type = problem_type
        self.selected_models = selected_models
        self.params = params

    def run(self):
        try:
            self.progress.emit("Dividindo dados...")
            X, y = separate_features_target(self.df, self.target_column)
            X_train, X_test, y_train, y_test = split_data(
                X, y, self.params['test_size'], self.params['seed'], self.problem_type
            )
            n_classes = y_train.nunique() if self.problem_type == "Classification" else 0

            dataset_info = {
                "filename": self.params['filename'],
                "target_column": self.target_column,
                "total_samples": len(self.df),
                "total_features": len(X.columns)
            }
            if self.problem_type == "Classification":
                dataset_info["class_distribution_train"] = y_train.value_counts().to_dict()
                dataset_info["class_distribution_test"] = y_test.value_counts().to_dict()
            else:
                dataset_info["target_stats"] = {
                    "mean": float(y_train.mean()),
                    "std": float(y_train.std()),
                    "min": float(y_train.min()),
                    "max": float(y_train.max())
                }
            
            all_results = {}
            for idx, instance in enumerate(self.selected_models, start=1):
                model_name = instance["model_name"]
                model_params = instance["params"]
                run_name = f"{model_name} #{instance['id']}"
                if model_name == "Logistic Regression" and n_classes >= 3:
                    solver = model_params.get("solver", "lbfgs")
                    penalty = model_params.get("penalty", "l2")
                    if solver == "liblinear":
                        if penalty in ("l1", "elasticnet"):
                            model_params["solver"] = "saga"
                        else:
                            model_params["solver"] = "lbfgs"

                self.progress.emit(f"Treinando {run_name}...")
                start_time = time.perf_counter()
                pipeline = build_pipeline(
                    self.params['scaler'], self.params['sampler'], 
                    model_name, model_params, self.problem_type, self.params['seed']
                )
                res = evaluate_model(pipeline, X_train, y_train, X_test, y_test, self.problem_type)
                duration_sec = time.perf_counter() - start_time
                
                self.progress.emit(f"Gerando gráficos para {run_name}...")
                plots = generate_plots(y_test, res["y_pred"], res["y_score"], self.problem_type)
                
                report = generate_report(
                    dataset_info,
                    {"test_size": self.params['test_size'], "scaler_name": self.params['scaler'], "sampler_name": self.params['sampler'], "model_name": run_name},
                    model_params,
                    res["metrics"], self.problem_type, self.params['seed']
                )
                
                all_results[run_name] = {
                    "metrics": res["metrics"],
                    "pipeline": res["pipeline"],
                    "plots": plots,
                    "report": report,
                    "params": model_params,
                    "duration_sec": duration_sec,
                    "y_test": y_test,
                    "y_pred": res["y_pred"]
                }
            
            self.finished.emit(all_results)
        except Exception as e:
            self.error.emit(str(e))

# --- Janela Principal ---
class MLApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ML Reprodutível Desktop v3 (Otimizado)")
        self.setGeometry(100, 100, 1200, 850)
        
        self.df = None
        self.target_column = None
        self.problem_type = None
        self.current_filename = ""
        self.selected_models = {}
        self.model_param_widgets = {}
        self.current_model_specs = {}
        self.model_editors = {}
        self.current_editor_model = None
        self.model_instances = []
        self.model_instance_counter = 1
        self.current_instance_id = None
        self.is_dark_theme = False
        self.last_metrics = []
        self.history_records = []
        self.history_path = os.path.join(os.path.dirname(__file__), "history.json")
        
        self.init_ui()
        self.load_history()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # --- Painel Lateral ---
        sidebar = QWidget()
        sidebar.setFixedWidth(320)
        sidebar_layout = QVBoxLayout(sidebar)
        
        # 1. Dados
        group_data = QGroupBox("1. Dados (CSV ou Parquet)")
        data_layout = QVBoxLayout()
        self.btn_load = QPushButton("Carregar Arquivo")
        self.btn_load.clicked.connect(self.load_file)
        self.lbl_file = QLabel("Nenhum arquivo carregado")
        self.lbl_file.setWordWrap(True)
        self.combo_target = QComboBox()
        self.combo_target.currentIndexChanged.connect(self.target_changed)
        data_layout.addWidget(self.btn_load)
        data_layout.addWidget(self.lbl_file)
        data_layout.addWidget(QLabel("Variável Target:"))
        data_layout.addWidget(self.combo_target)
        group_data.setLayout(data_layout)
        
        # 2. Parâmetros
        group_params = QGroupBox("2. Parâmetros Metodológicos")
        params_layout = QVBoxLayout()
        params_layout.addWidget(QLabel("Proporção Teste (%):"))
        self.spin_test = QSpinBox()
        self.spin_test.setRange(10, 50); self.spin_test.setValue(20)
        params_layout.addWidget(self.spin_test)
        
        params_layout.addWidget(QLabel("Seed Fixa:"))
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 9999); self.spin_seed.setValue(42)
        params_layout.addWidget(self.spin_seed)
        
        params_layout.addWidget(QLabel("Scaler:"))
        self.combo_scaler = QComboBox(); self.combo_scaler.addItems(SCALER_OPTIONS)
        params_layout.addWidget(self.combo_scaler)
        
        params_layout.addWidget(QLabel("Sampler (Classificação):"))
        self.combo_sampler = QComboBox(); self.combo_sampler.addItems(SAMPLER_OPTIONS)
        params_layout.addWidget(self.combo_sampler)
        group_params.setLayout(params_layout)
        
        # 3. Modelos
        group_models = QGroupBox("3. Modelos (Máx 5)")
        models_layout = QVBoxLayout()
        self.lbl_selected_models = QLabel("Selecionados: 0/5")
        self.list_models = QListWidget()
        self.list_models.currentItemChanged.connect(self.model_selected)
        models_layout.addWidget(self.lbl_selected_models)
        models_layout.addWidget(self.list_models)
        group_models.setLayout(models_layout)
        
        # Status e Progresso
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.lbl_status = QLabel("Pronto")
        
        self.btn_run = QPushButton(" Executar Experimento")
        self.btn_run.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; height: 45px;")
        self.btn_run.clicked.connect(self.run_experiment)
        
        group_models.setSizePolicy(group_models.sizePolicy().horizontalPolicy(), group_models.sizePolicy().verticalPolicy())
        sidebar_layout.addWidget(group_data)
        sidebar_layout.addWidget(group_params)
        sidebar_layout.addWidget(group_models, 1)
        sidebar_layout.addWidget(self.lbl_status)
        sidebar_layout.addWidget(self.progress_bar)
        sidebar_layout.addWidget(self.btn_run)
        sidebar_layout.addStretch()
        
        # --- Painel Principal ---
        self.tabs = QTabWidget()
        self.welcome_tab = QWidget()
        welcome_layout = QVBoxLayout(self.welcome_tab)
        header_row = QHBoxLayout()
        header_row.addWidget(QLabel("<h1>trAIn V.5</h1>"))
        header_row.addStretch()
        theme_wrap = QHBoxLayout()
        theme_wrap.setContentsMargins(0, 0, 0, 0)
        theme_label = QLabel("Tema")
        self.theme_toggle = ToggleSwitch()
        self.theme_toggle.setToolTip("Alternar tema claro/escuro")
        self.theme_toggle.toggled.connect(self.set_theme)
        theme_wrap.addWidget(theme_label)
        theme_wrap.addWidget(self.theme_toggle)
        theme_container = QWidget()
        theme_container.setLayout(theme_wrap)
        header_row.addWidget(theme_container)
        welcome_layout.addLayout(header_row)
        welcome_layout.addWidget(QLabel("Suporte a <b>Parquet</b> e processamento em <b>segundo plano</b> para arquivos grandes."))
        welcome_layout.addWidget(QLabel("<h3>Modelos selecionados</h3>"))
        self.list_selected_models = QListWidget()
        self.list_selected_models.currentItemChanged.connect(self.selected_instance_changed)
        welcome_layout.addWidget(self.list_selected_models)
        welcome_layout.addWidget(QLabel("<h3>Parametros do modelo</h3>"))
        self.model_editor_stack = QStackedWidget()
        self.model_editor_placeholder = QLabel("Selecione um modelo na lista para editar os parametros.")
        self.model_editor_placeholder.setWordWrap(True)
        placeholder_wrap = QWidget()
        placeholder_layout = QVBoxLayout(placeholder_wrap)
        placeholder_layout.addWidget(self.model_editor_placeholder)
        placeholder_layout.addStretch()
        self.model_editor_stack.addWidget(placeholder_wrap)
        welcome_layout.addWidget(self.model_editor_stack, 1)
        welcome_layout.addStretch()
        self.tabs.addTab(self.welcome_tab, "Início")

        self.history_tab = QWidget()
        history_layout = QVBoxLayout(self.history_tab)
        history_header = QHBoxLayout()
        history_header.addWidget(QLabel("<h2>Historico de Execucoes</h2>"))
        history_header.addStretch()
        btn_export_history = QPushButton("Exportar Historico")
        btn_export_history.clicked.connect(self.export_history)
        btn_clear_history = QPushButton("Limpar Historico")
        btn_clear_history.clicked.connect(self.clear_history)
        history_header.addWidget(btn_export_history)
        history_header.addWidget(btn_clear_history)
        history_layout.addLayout(history_header)

        self.history_table = QTableWidget()
        self.history_table.setSizeAdjustPolicy(QAbstractItemView.SizeAdjustPolicy.AdjustToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        history_layout.addWidget(self.history_table)
        self.tabs.addTab(self.history_tab, "Historico")
        
        main_layout.addWidget(sidebar)

        self.right_container = QWidget()
        self.right_layout = QVBoxLayout(self.right_container)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.addWidget(self.tabs)
        main_layout.addWidget(self.right_container)

        self.literature_panel = LiteraturePanel(self.right_container)
        reserved = self.literature_panel.handle.height()
        self.right_layout.setContentsMargins(0, 0, 0, reserved)
        self.literature_panel.setFixedHeight(reserved)
        self.literature_panel.update_position()
        self.literature_panel.raise_()
        self.apply_theme()

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Abrir Dados", "", "Dados (*.csv *.parquet *.pqt)")
        if file_path:
            self.lbl_status.setText("Carregando arquivo...")
            self.btn_load.setEnabled(False)
            try:
                self.df = load_data(file_path)
                self.current_filename = os.path.basename(file_path)
                self.lbl_file.setText(f"Arquivo: {self.current_filename}")
                self.combo_target.clear()
                self.combo_target.addItems([None] + self.df.columns.tolist())
                self.lbl_status.setText("Arquivo carregado.")
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Falha ao carregar: {e}")
                self.lbl_status.setText("Erro no carregamento.")
            finally:
                self.btn_load.setEnabled(True)

    def target_changed(self):
        target = self.combo_target.currentText()
        if target and self.df is not None:
            self.target_column = target
            self.problem_type = identify_problem_type(self.df[target])
            self.selected_models = {}
            self.model_param_widgets = {}
            self.current_model_specs = {spec.name: spec for spec in get_model_specs_by_problem(self.problem_type)}
            self.model_editors = {}
            self.current_editor_model = None
            self.model_instances = []
            self.model_instance_counter = 1
            self.current_instance_id = None
            self.list_models.clear()
            self.list_models.addItems(self.current_model_specs.keys())
            self.reset_model_editor_stack()
            self.refresh_selected_models_view()
            self.combo_sampler.setEnabled(self.problem_type == "Classification")

    def run_experiment(self):
        if not self.model_instances or len(self.model_instances) > 5:
            QMessageBox.warning(self, "Aviso", "Selecione entre 1 e 5 modelos.")
            return
        
        params = {
            'test_size': self.spin_test.value() / 100.0,
            'seed': self.spin_seed.value(),
            'scaler': self.combo_scaler.currentText(),
            'sampler': self.combo_sampler.currentText() if self.problem_type == "Classification" else "None",
            'filename': self.current_filename
        }
        
        # Configurar Worker Thread
        self.worker = MLWorker(self.df, self.target_column, self.problem_type, self.model_instances, params)
        self.worker.progress.connect(self.update_status)
        self.worker.finished.connect(self.display_results)
        self.worker.error.connect(self.handle_error)
        
        # UI Feedback
        self.btn_run.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0) # Modo indeterminado
        self.worker.start()

    def update_status(self, msg):
        self.lbl_status.setText(msg)

    def handle_error(self, msg):
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Erro no Processamento", msg)
        self.lbl_status.setText("Erro.")

    def display_results(self, results):
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("Concluído!")

        for i in reversed(range(self.tabs.count())):
            widget = self.tabs.widget(i)
            if widget not in (self.welcome_tab, self.history_tab):
                self.tabs.removeTab(i)
        
        # Aba Comparação
        comp_tab = QWidget()
        comp_layout = QVBoxLayout(comp_tab)
        comp_header = QHBoxLayout()
        comp_header.addWidget(QLabel("<h2>Comparacao de Metricas</h2>"))
        comp_header.addStretch()
        btn_export_results = QPushButton("Exportar Resultados")
        btn_export_results.clicked.connect(self.export_results)
        comp_header.addWidget(btn_export_results)
        comp_layout.addLayout(comp_header)

        table = QTableWidget()
        table.setSizeAdjustPolicy(QAbstractItemView.SizeAdjustPolicy.AdjustToContents)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        comp_layout.addWidget(table)
        self.tabs.addTab(comp_tab, "Comparação")
        
        metrics_list = []
        for m_name, m_res in results.items():
            # Aba Individual
            model_tab = QWidget()
            tab_layout = QVBoxLayout(model_tab)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            content = QWidget()
            model_layout = QVBoxLayout(content)
            
            report_html = self.build_report_html(m_name, m_res["report"], m_res["plots"])
            txt = QTextEdit(); txt.setHtml(report_html); txt.setReadOnly(True)
            btn_exp = QPushButton(f"Exportar Pipeline ({m_name})")
            btn_exp.clicked.connect(lambda ch, p=m_res["pipeline"], n=m_name: self.export_pipeline(p, n))
            model_layout.addWidget(txt)
            model_layout.addWidget(btn_exp)
            scroll.setWidget(content)
            tab_layout.addWidget(scroll)
            self.tabs.addTab(model_tab, m_name)
            
            m_data = {"Modelo": m_name}; m_data.update(m_res["metrics"])
            metrics_list.append(m_data)

            history_entry = {
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "modelo": m_name,
                "params": m_res.get("params", {}),
                "tempo_sec": float(m_res.get("duration_sec", 0.0)),
                "metricas": m_res.get("metrics", {})
            }
            self.history_records.append(history_entry)
            
        # Preencher Tabela
        if metrics_list:
            self.last_metrics = metrics_list
            headers = list(metrics_list[0].keys())
            table.setColumnCount(len(headers)); table.setRowCount(len(metrics_list))
            table.setHorizontalHeaderLabels(headers)
            for r, data in enumerate(metrics_list):
                for c, k in enumerate(headers):
                    v = data[k]
                    table.setItem(r, c, QTableWidgetItem(f"{v:.4f}" if isinstance(v, float) else str(v)))
        
        self.tabs.setCurrentIndex(1)
        self.update_history_table()
        self.save_history()

    def export_results(self):
        if not self.last_metrics:
            QMessageBox.warning(self, "Aviso", "Nenhum resultado para exportar.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Salvar Resultados", "resultados.csv", "CSV (*.csv)"
        )
        if not path:
            return

        df = pd.DataFrame(self.last_metrics)
        df.to_csv(path, index=False)
        QMessageBox.information(self, "Sucesso", "Resultados exportados!")

    def export_history(self):
        if not self.history_records:
            QMessageBox.warning(self, "Aviso", "Historico vazio.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Salvar Historico", "historico.json", "JSON (*.json)"
        )
        if not path:
            return

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.history_records, handle, ensure_ascii=True, indent=2)
        QMessageBox.information(self, "Sucesso", "Historico exportado!")

    def clear_history(self):
        self.history_records = []
        self.update_history_table()
        self.save_history()

    def load_history(self):
        if not os.path.exists(self.history_path):
            return
        try:
            with open(self.history_path, "r", encoding="utf-8") as handle:
                self.history_records = json.load(handle) or []
        except (json.JSONDecodeError, OSError):
            self.history_records = []
        self.update_history_table()

    def save_history(self):
        try:
            with open(self.history_path, "w", encoding="utf-8") as handle:
                json.dump(self.history_records, handle, ensure_ascii=True, indent=2)
        except OSError:
            pass

    def update_history_table(self):
        headers = ["timestamp", "modelo", "tempo_sec", "params", "metricas"]
        self.history_table.setColumnCount(len(headers))
        self.history_table.setHorizontalHeaderLabels(headers)
        self.history_table.setRowCount(len(self.history_records))
        self.history_table.setWordWrap(True)

        for r, item in enumerate(self.history_records):
            values = [
                item.get("timestamp", ""),
                item.get("modelo", ""),
                f"{item.get('tempo_sec', 0.0):.4f}",
                self._dict_to_multiline(item.get("params", {})),
                self._dict_to_multiline(item.get("metricas", {}))
            ]
            for c, value in enumerate(values):
                self.history_table.setItem(r, c, QTableWidgetItem(str(value)))
        self.history_table.resizeRowsToContents()

    def _dict_to_multiline(self, data: dict) -> str:
        if not data:
            return ""
        return "\n".join([f"{k}: {v}" for k, v in data.items()])

    def build_report_html(self, title: str, report_text: str, plots: dict) -> str:
        safe_report = html.escape(report_text)
        parts = [
            f"<h2>{html.escape(title)}</h2>",
            "<pre style='font-family: Consolas, monospace; font-size: 12px; white-space: pre-wrap;'>",
            safe_report,
            "</pre>"
        ]

        if plots:
            parts.append("<h3>Graficos</h3>")
            for fig in plots.values():
                uri = self.figure_to_data_uri(fig)
                parts.append(
                    "<div style='margin: 12px 0;'>"
                    f"<img src='{uri}' style='max-width: 100%; height: auto;'/>"
                    "</div>"
                )
        else:
            parts.append("<p>Sem graficos disponiveis.</p>")

        return "".join(parts)

    def figure_to_data_uri(self, fig) -> str:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def export_pipeline(self, pipeline, model_name):
        path, _ = QFileDialog.getSaveFileName(self, "Salvar", f"pipeline_{model_name.lower()}.joblib", "Joblib (*.joblib)")
        if path: joblib.dump(pipeline, path); QMessageBox.information(self, "Sucesso", "Salvo!")

    def create_model_page(self, spec):
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QFormLayout()
        widgets = {}
        existing = {}

        for param in spec.params:
            if param.kind == "int":
                widget = QSpinBox()
                widget.setRange(int(param.min or 0), int(param.max or 999999))
                if param.step:
                    widget.setSingleStep(int(param.step))
                widget.setValue(int(existing.get(param.name, param.default)))
            elif param.kind == "float":
                widget = QDoubleSpinBox()
                widget.setRange(float(param.min or 0.0), float(param.max or 999999.0))
                widget.setDecimals(4)
                if param.step:
                    widget.setSingleStep(float(param.step))
                widget.setValue(float(existing.get(param.name, param.default)))
            elif param.kind == "bool":
                widget = QComboBox()
                widget.addItems(["True", "False"])
                value = existing.get(param.name, param.default)
                widget.setCurrentText("True" if value else "False")
            elif param.kind == "choice":
                widget = QComboBox()
                widget.addItems(param.choices or [])
                widget.setCurrentText(str(existing.get(param.name, param.default)))
            else:
                widget = QLineEdit()
                widget.setText(str(existing.get(param.name, param.default)))

            widgets[param.name] = widget
            form.addRow(QLabel(param.label), widget)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("Adicionar")
        btn_update = QPushButton("Atualizar")
        btn_remove = QPushButton("Remover")
        btn_add.clicked.connect(lambda _ch, name=spec.name: self.add_model(name))
        btn_update.clicked.connect(lambda _ch, name=spec.name: self.update_model_instance(name))
        btn_remove.clicked.connect(lambda _ch, name=spec.name: self.remove_model(name))
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_update)
        btn_row.addWidget(btn_remove)

        layout.addLayout(form)
        layout.addLayout(btn_row)
        layout.addStretch()

        self.model_param_widgets[spec.name] = widgets
        return page

    def add_model(self, model_name):
        if model_name not in self.current_model_specs:
            return

        if len(self.model_instances) >= 5:
            QMessageBox.warning(self, "Aviso", "Limite de 5 modelos selecionados.")
            return

        params = self.collect_params(model_name)
        instance = {
            "id": self.model_instance_counter,
            "model_name": model_name,
            "params": params
        }
        self.model_instance_counter += 1
        self.model_instances.append(instance)
        self.refresh_selected_models_view()

    def remove_model(self, model_name):
        if self.current_instance_id is None:
            QMessageBox.warning(self, "Aviso", "Selecione um modelo na lista para remover.")
            return

        self.model_instances = [m for m in self.model_instances if m["id"] != self.current_instance_id]
        self.current_instance_id = None
        self.refresh_selected_models_view()
        self.list_selected_models.clearSelection()

    def update_model_instance(self, model_name):
        if self.current_instance_id is None:
            QMessageBox.warning(self, "Aviso", "Selecione um modelo na lista para atualizar.")
            return

        params = self.collect_params(model_name)
        updated = False
        for instance in self.model_instances:
            if instance["id"] == self.current_instance_id:
                instance["params"] = params
                instance["model_name"] = model_name
                updated = True
                break

        if not updated:
            QMessageBox.warning(self, "Aviso", "Nao foi possivel localizar o modelo selecionado.")
            return

        self.refresh_selected_models_view()

    def collect_params(self, model_name):
        spec = self.current_model_specs.get(model_name)
        widgets = self.model_param_widgets.get(model_name, {})
        params = {}
        if spec is None:
            return params

        for param in spec.params:
            widget = widgets.get(param.name)
            if widget is None:
                continue
            if param.kind == "int":
                params[param.name] = int(widget.value())
            elif param.kind == "float":
                params[param.name] = float(widget.value())
            elif param.kind == "bool":
                params[param.name] = widget.currentText() == "True"
            elif param.kind == "choice":
                params[param.name] = widget.currentText()
            else:
                params[param.name] = widget.text()

        return params

    def refresh_selected_models_view(self):
        self.list_selected_models.clear()
        for instance in self.model_instances:
            name = instance["model_name"]
            params = instance["params"]
            parts = [f"{k}={v}" for k, v in params.items()]
            line = f"{name} #{instance['id']} ({', '.join(parts)})" if parts else f"{name} #{instance['id']}"
            item = QListWidgetItem(line)
            item.setData(Qt.ItemDataRole.UserRole, instance["id"])
            self.list_selected_models.addItem(item)

        self.lbl_selected_models.setText(f"Selecionados: {len(self.model_instances)}/5")

    def model_selected(self, current, _previous=None):
        if current is None:
            self.current_editor_model = None
            self.model_editor_stack.setCurrentIndex(0)
            self.update_literature_panel(None)
            return

        model_name = current.text()
        self.current_editor_model = model_name
        self.show_model_editor(model_name)
        self.update_literature_panel(model_name)

    def selected_instance_changed(self, current, _previous=None):
        if current is None:
            self.current_instance_id = None
            self.update_literature_panel(None)
            return

        instance_id = current.data(Qt.ItemDataRole.UserRole)
        instance = next((m for m in self.model_instances if m["id"] == instance_id), None)
        if instance is None:
            self.current_instance_id = None
            return

        self.current_instance_id = instance_id
        model_name = instance["model_name"]
        self.current_editor_model = model_name
        self.show_model_editor(model_name)
        self.populate_model_editor(model_name, instance["params"])
        self.update_literature_panel(model_name)

    def update_literature_panel(self, model_name):
        if not hasattr(self, "literature_panel"):
            return
        model_label = model_name or ""
        html_text = get_literature_html(model_label, self.is_dark_theme)
        self.literature_panel.set_html(html_text)

    def show_model_editor(self, model_name):
        editor = self.model_editors.get(model_name)
        if editor is None:
            spec = self.current_model_specs.get(model_name)
            if spec is None:
                self.model_editor_stack.setCurrentIndex(0)
                return
            editor = self.create_model_page(spec)
            self.model_editors[model_name] = editor
            self.model_editor_stack.addWidget(editor)

        self.model_editor_stack.setCurrentWidget(editor)

    def populate_model_editor(self, model_name, params):
        widgets = self.model_param_widgets.get(model_name, {})
        for key, widget in widgets.items():
            value = params.get(key)
            if value is None:
                continue
            if isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QComboBox):
                widget.setCurrentText("True" if value is True else "False" if value is False else str(value))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))

    def reset_model_editor_stack(self):
        for i in reversed(range(self.model_editor_stack.count())):
            widget = self.model_editor_stack.widget(i)
            if widget is not self.model_editor_stack.widget(0):
                self.model_editor_stack.removeWidget(widget)
                widget.deleteLater()
        self.model_editor_stack.setCurrentIndex(0)

    def set_theme(self, is_dark):
        self.is_dark_theme = is_dark
        self.apply_theme()

    def apply_theme(self):
        app = QApplication.instance()
        if self.is_dark_theme:
            palette = QPalette()
            palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
            palette.setColor(QPalette.ColorRole.WindowText, QColor(230, 230, 230))
            palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
            palette.setColor(QPalette.ColorRole.AlternateBase, QColor(35, 35, 35))
            palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(230, 230, 230))
            palette.setColor(QPalette.ColorRole.ToolTipText, QColor(30, 30, 30))
            palette.setColor(QPalette.ColorRole.Text, QColor(230, 230, 230))
            palette.setColor(QPalette.ColorRole.Button, QColor(40, 40, 40))
            palette.setColor(QPalette.ColorRole.ButtonText, QColor(230, 230, 230))
            palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 255, 255))
            palette.setColor(QPalette.ColorRole.Highlight, QColor(46, 125, 50))
            palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
            if app:
                app.setPalette(palette)
            self.setStyleSheet(
                "QWidget { color: #e6e6e6; }"
                "QGroupBox { border: 1px solid #3a3a3a; border-radius: 8px; margin-top: 10px; padding-top: 18px; }"
                "QGroupBox::title {"
                "  subcontrol-origin: margin; left: 12px; top: 0px;"
                "  background: #1e1e1e; padding: 1px 8px; color: #cfcfcf;"
                "  border: 1px solid #3a3a3a; border-radius: 6px;"
                "}"
                "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QListWidget, QTableWidget {"
                "  background-color: #1f1f1f; color: #e6e6e6; border: 1px solid #3a3a3a;"
                "  border-radius: 6px; padding: 4px;"
                "}"
                "QComboBox::drop-down { border: none; width: 22px; }"
                "QComboBox QAbstractItemView { background: #1f1f1f; color: #e6e6e6; selection-background-color: #2e7d32; }"
                "QTabWidget::pane { border: 1px solid #3a3a3a; border-radius: 6px; }"
                "QTabBar::tab { background: #2b2b2b; color: #e6e6e6; padding: 6px 12px; border-top-left-radius: 6px; border-top-right-radius: 6px; }"
                "QTabBar::tab:selected { background: #3a3a3a; }"
                "QPushButton { background: #2b2d30; color: #e6e6e6; border: 1px solid #3a3a3a; border-radius: 8px; padding: 6px 12px; }"
                "QPushButton:hover { background: #3a3a3a; }"
                "QPushButton:pressed { background: #262626; }"
                "QPushButton:disabled { background: #252525; color: #777777; border-color: #2b2b2b; }"
                "QHeaderView::section { background: #2b2b2b; color: #e6e6e6; border: 1px solid #3a3a3a; padding: 6px; }"
                "QTableWidget { gridline-color: #3a3a3a; }"
                "QScrollBar:vertical { background: #1f1f1f; width: 12px; margin: 2px; }"
                "QScrollBar::handle:vertical { background: #3a3a3a; border-radius: 6px; min-height: 24px; }"
                "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }"
                "#literaturePanel { background: #1b1b1b; border-top: 1px solid #3a3a3a; }"
                "#literatureHandle { background: #2b2b2b; border-top: 1px solid #3a3a3a; }"
                "#literatureHandle QLabel { color: #e6e6e6; font-weight: bold; }"
                "#literatureViewer { background: #1f1f1f; color: #e6e6e6; border: 1px solid #3a3a3a; border-radius: 6px; padding: 6px; }"
            )
        else:
            if app:
                app.setPalette(app.style().standardPalette())
            self.setStyleSheet(
                "QWidget { color: #1f1f1f; }"
                "QGroupBox { border: 1px solid #d6d3cf; border-radius: 8px; margin-top: 10px; padding-top: 18px; background: #f4f1ec; }"
                "QGroupBox::title {"
                "  subcontrol-origin: margin; left: 12px; top: 0px;"
                "  background: #f4f1ec; padding: 1px 8px; color: #3b3b3b;"
                "  border: 1px solid #d6d3cf; border-radius: 6px;"
                "}"
                "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QListWidget, QTableWidget {"
                "  background-color: #ffffff; color: #1f1f1f; border: 1px solid #d6d3cf;"
                "  border-radius: 6px; padding: 4px;"
                "}"
                "QComboBox::drop-down { border: none; width: 22px; }"
                "QComboBox QAbstractItemView { background: #ffffff; color: #1f1f1f; selection-background-color: #2e7d32; selection-color: #ffffff; }"
                "QTabWidget::pane { border: 1px solid #d6d3cf; border-radius: 6px; background: #ffffff; }"
                "QTabBar::tab { background: #e8e3dc; color: #1f1f1f; padding: 6px 12px; border-top-left-radius: 6px; border-top-right-radius: 6px; }"
                "QTabBar::tab:selected { background: #ffffff; }"
                "QPushButton { background: #ede8e1; color: #1f1f1f; border: 1px solid #d6d3cf; border-radius: 8px; padding: 6px 12px; }"
                "QPushButton:hover { background: #f4efe9; }"
                "QPushButton:pressed { background: #e3ddd6; }"
                "QPushButton:disabled { background: #f6f3ef; color: #9a948d; border-color: #e4dfd7; }"
                "QHeaderView::section { background: #e8e3dc; color: #1f1f1f; border: 1px solid #d6d3cf; padding: 6px; }"
                "QTableWidget { gridline-color: #d6d3cf; }"
                "QScrollBar:vertical { background: #f4f1ec; width: 12px; margin: 2px; }"
                "QScrollBar::handle:vertical { background: #d6d3cf; border-radius: 6px; min-height: 24px; }"
                "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }"
                "#literaturePanel { background: #ffffff; border-top: 1px solid #d6d3cf; }"
                "#literatureHandle { background: #e8e3dc; border-top: 1px solid #d6d3cf; }"
                "#literatureHandle QLabel { color: #1f1f1f; font-weight: bold; }"
                "#literatureViewer { background: #ffffff; color: #1f1f1f; border: 1px solid #d6d3cf; border-radius: 6px; padding: 6px; }"
            )

        if hasattr(self, "literature_panel"):
            self.literature_panel.update_position()
            current_name = self.current_editor_model or ""
            self.literature_panel.set_html(get_literature_html(current_name, self.is_dark_theme))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "literature_panel"):
            self.literature_panel.update_position()

    def showEvent(self, event):
        super().showEvent(event)
        if hasattr(self, "literature_panel"):
            QTimer.singleShot(0, self.literature_panel.update_position)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MLApp(); window.show()
    sys.exit(app.exec())
