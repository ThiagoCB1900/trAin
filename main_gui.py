import sys
import pandas as pd
import numpy as np
import io
import joblib
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QFileDialog, QComboBox, QLabel, QSpinBox, 
    QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit, 
    QMessageBox, QListWidget, QGroupBox, QProgressBar,
    QFormLayout, QDoubleSpinBox, QLineEdit, QStackedWidget
)
from PyQt6.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Importando lógica existente
from data_handler import load_data, separate_features_target, identify_problem_type, split_data
from pipeline_builder import build_pipeline, SCALER_OPTIONS, SAMPLER_OPTIONS
from experiment_runner import evaluate_model, generate_plots, save_pipeline_to_buffer
from reporter import generate_report
from models.registry import get_model_specs_by_problem

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
            
            all_results = {}
            for model_name, model_params in self.selected_models.items():
                if model_name == "Logistic Regression" and n_classes >= 3:
                    solver = model_params.get("solver", "lbfgs")
                    penalty = model_params.get("penalty", "l2")
                    if solver == "liblinear":
                        if penalty in ("l1", "elasticnet"):
                            model_params["solver"] = "saga"
                        else:
                            model_params["solver"] = "lbfgs"

                self.progress.emit(f"Treinando {model_name}...")
                pipeline = build_pipeline(
                    self.params['scaler'], self.params['sampler'], 
                    model_name, model_params, self.problem_type, self.params['seed']
                )
                res = evaluate_model(pipeline, X_train, y_train, X_test, y_test, self.problem_type)
                
                self.progress.emit(f"Gerando gráficos para {model_name}...")
                plots = generate_plots(y_test, res["y_pred"], res["y_score"], self.problem_type)
                
                report = generate_report(
                    {"filename": self.params['filename'], "target_column": self.target_column, "total_samples": len(self.df), "total_features": len(X.columns)},
                    {"test_size": self.params['test_size'], "scaler_name": self.params['scaler'], "sampler_name": self.params['sampler'], "model_name": model_name},
                    res["metrics"], self.problem_type, self.params['seed']
                )
                
                all_results[model_name] = {
                    "metrics": res["metrics"],
                    "pipeline": res["pipeline"],
                    "plots": plots,
                    "report": report,
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
        
        self.init_ui()
        
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
        group_models = QGroupBox("3. Modelos (Máx 3)")
        models_layout = QVBoxLayout()
        self.lbl_selected_models = QLabel("Selecionados: 0/3")
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
        welcome_layout.addWidget(QLabel("<h1>ML Reprodutível v3</h1>"))
        welcome_layout.addWidget(QLabel("Suporte a <b>Parquet</b> e processamento em <b>segundo plano</b> para arquivos grandes."))
        welcome_layout.addWidget(QLabel("<h3>Modelos selecionados</h3>"))
        self.list_selected_models = QListWidget()
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
        
        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.tabs)

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
            self.list_models.clear()
            self.list_models.addItems(self.current_model_specs.keys())
            self.reset_model_editor_stack()
            self.refresh_selected_models_view()
            self.combo_sampler.setEnabled(self.problem_type == "Classification")

    def run_experiment(self):
        if not self.selected_models or len(self.selected_models) > 3:
            QMessageBox.warning(self, "Aviso", "Selecione entre 1 e 3 modelos.")
            return
        
        params = {
            'test_size': self.spin_test.value() / 100.0,
            'seed': self.spin_seed.value(),
            'scaler': self.combo_scaler.currentText(),
            'sampler': self.combo_sampler.currentText() if self.problem_type == "Classification" else "None",
            'filename': self.current_filename
        }
        
        # Configurar Worker Thread
        self.worker = MLWorker(self.df, self.target_column, self.problem_type, self.selected_models, params)
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
        
        while self.tabs.count() > 1: self.tabs.removeTab(1)
        
        # Aba Comparação
        comp_tab = QWidget(); comp_layout = QVBoxLayout(comp_tab)
        table = QTableWidget(); comp_layout.addWidget(QLabel("<h2>Comparação de Métricas</h2>")); comp_layout.addWidget(table)
        self.tabs.addTab(comp_tab, "Comparação")
        
        metrics_list = []
        for m_name, m_res in results.items():
            # Aba Individual
            model_tab = QWidget(); model_layout = QHBoxLayout(model_tab)
            left = QVBoxLayout(); right = QVBoxLayout()
            
            txt = QTextEdit(); txt.setPlainText(m_res["report"]); txt.setReadOnly(True)
            btn_exp = QPushButton(f"Exportar Pipeline ({m_name})")
            btn_exp.clicked.connect(lambda ch, p=m_res["pipeline"], n=m_name: self.export_pipeline(p, n))
            left.addWidget(QLabel(f"<h3>{m_name}</h3>")); left.addWidget(txt); left.addWidget(btn_exp)
            
            if m_res["plots"]:
                for fig in m_res["plots"].values(): right.addWidget(FigureCanvas(fig))
            else: right.addWidget(QLabel("Sem gráficos disponíveis."))
            
            model_layout.addLayout(left, 1); model_layout.addLayout(right, 2)
            self.tabs.addTab(model_tab, m_name)
            
            m_data = {"Modelo": m_name}; m_data.update(m_res["metrics"])
            metrics_list.append(m_data)
            
        # Preencher Tabela
        if metrics_list:
            headers = list(metrics_list[0].keys())
            table.setColumnCount(len(headers)); table.setRowCount(len(metrics_list))
            table.setHorizontalHeaderLabels(headers)
            for r, data in enumerate(metrics_list):
                for c, k in enumerate(headers):
                    v = data[k]
                    table.setItem(r, c, QTableWidgetItem(f"{v:.4f}" if isinstance(v, float) else str(v)))
        
        self.tabs.setCurrentIndex(1)

    def export_pipeline(self, pipeline, model_name):
        path, _ = QFileDialog.getSaveFileName(self, "Salvar", f"pipeline_{model_name.lower()}.joblib", "Joblib (*.joblib)")
        if path: joblib.dump(pipeline, path); QMessageBox.information(self, "Sucesso", "Salvo!")

    def create_model_page(self, spec):
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QFormLayout()
        widgets = {}
        existing = self.selected_models.get(spec.name, {})

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
        btn_add = QPushButton("Adicionar/Atualizar")
        btn_remove = QPushButton("Remover")
        btn_add.clicked.connect(lambda _ch, name=spec.name: self.add_model(name))
        btn_remove.clicked.connect(lambda _ch, name=spec.name: self.remove_model(name))
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_remove)

        layout.addLayout(form)
        layout.addLayout(btn_row)
        layout.addStretch()

        self.model_param_widgets[spec.name] = widgets
        return page

    def add_model(self, model_name):
        if model_name not in self.current_model_specs:
            return

        if model_name not in self.selected_models and len(self.selected_models) >= 3:
            QMessageBox.warning(self, "Aviso", "Limite de 3 modelos selecionados.")
            return

        self.selected_models[model_name] = self.collect_params(model_name)
        self.refresh_selected_models_view()

    def remove_model(self, model_name):
        if model_name in self.selected_models:
            del self.selected_models[model_name]
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
        for name, params in self.selected_models.items():
            parts = [f"{k}={v}" for k, v in params.items()]
            line = f"{name} ({', '.join(parts)})" if parts else name
            self.list_selected_models.addItem(line)

        self.lbl_selected_models.setText(f"Selecionados: {len(self.selected_models)}/3")

    def model_selected(self, current, _previous=None):
        if current is None:
            self.current_editor_model = None
            self.model_editor_stack.setCurrentIndex(0)
            return

        model_name = current.text()
        self.current_editor_model = model_name
        self.show_model_editor(model_name)

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

    def reset_model_editor_stack(self):
        for i in reversed(range(self.model_editor_stack.count())):
            widget = self.model_editor_stack.widget(i)
            if widget is not self.model_editor_stack.widget(0):
                self.model_editor_stack.removeWidget(widget)
                widget.deleteLater()
        self.model_editor_stack.setCurrentIndex(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MLApp(); window.show()
    sys.exit(app.exec())
