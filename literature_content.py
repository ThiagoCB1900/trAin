from pathlib import Path


MODEL_LITERATURE_FILES = {
    "RandomForestClassifier": "random_forest/index.html",
    "RandomForestRegressor": "random_forest_regressor/index.html",
    "Logistic Regression": "logistic_regression/index.html",
    "Linear Regression": "linear_regression/index.html",
    "Ridge Regression": "ridge_regression/index.html",
    "SVR": "svr/index.html",
    "KNN": "knn/index.html",
    "Naive Bayes": "naive_bayes/index.html",
    "Gradient Boosting": "gradient_boosting/index.html",
    "Decision Tree": "decision_tree/index.html",
    "Decision Tree Regressor": "decision_tree_regressor/index.html",
    "Gradient Boosting Regressor": "gradient_boosting_regressor/index.html",
    "SVM": "svm/index.html",
    "XGBoost": "xgboost/index.html",
    "XGBoost Regressor": "xgboost_regressor/index.html",
}


def _wrap_html(body_html: str, is_dark: bool) -> str:
    colors = {
        "bg": "#14212b" if is_dark else "#ffffff",
        "text": "#e7f2ef" if is_dark else "#1f2d2a",
        "muted": "#a9c1bc" if is_dark else "#5f7370",
        "accent": "#49b9a6" if is_dark else "#2f8f83",
        "accent_hover": "#5ecab7" if is_dark else "#3ca094",
        "card": "#1a2b38" if is_dark else "#f4f8f7",
        "border": "#2d4152" if is_dark else "#c8dbd5",
        "callout_bg": "#1f4b4d" if is_dark else "#d8efea",
        "warning_bg": "#4d2020" if is_dark else "#ffe8e8",
        "code_bg": "#0f1720" if is_dark else "#edf5f2",
    }
    return (
        "<html><head><style>"
        "* { color: " + colors["text"] + " !important; }"
        "body { "
        f"  font-family: 'Segoe UI', Arial, sans-serif !important; "
        f"  color: {colors['text']} !important; "
        f"  background: {colors['bg']} !important; "
        "  line-height: 1.7 !important; "
        "  padding: 4px !important; "
        "}"
        "h1, h2, h3, h4, h5, h6 { "
        f"  color: {colors['text']} !important; "
        "  font-weight: 700 !important; "
        "}"
        "a { "
        f"  color: {colors['accent']} !important; "
        "  text-decoration: none !important; "
        "}"
        "a:hover { "
        f"  color: {colors['accent_hover']} !important; "
        "  text-decoration: underline !important; "
        "}"
        "code, pre, .formula { "
        f"  background: {colors['code_bg']} !important; "
        f"  border: 1px solid {colors['border']} !important; "
        f"  color: {colors['text']} !important; "
        "  border-radius: 6px !important; "
        "  padding: 4px 6px !important; "
        "}"
        ".card, .formula-box { "
        f"  border: 1px solid {colors['border']} !important; "
        f"  background: {colors['card']} !important; "
        "  border-radius: 10px !important; "
        "  padding: 12px !important; "
        "}"
        ".callout { "
        f"  border-left: 4px solid {colors['accent']} !important; "
        f"  background: {colors['callout_bg']} !important; "
        f"  color: {colors['text']} !important; "
        "  border-radius: 8px !important; "
        "  padding: 12px !important; "
        "}"
        ".warning { "
        f"  border-left: 4px solid #d16b6b !important; "
        f"  background: {colors['warning_bg']} !important; "
        f"  color: {colors['text']} !important; "
        "  border-radius: 8px !important; "
        "  padding: 12px !important; "
        "}"
        "table { "
        f"  border-collapse: collapse !important; "
        "  width: 100% !important; "
        "  margin: 12px 0 !important; "
        "}"
        "table th, table td { "
        f"  border: 1px solid {colors['border']} !important; "
        f"  color: {colors['text']} !important; "
        "  padding: 10px !important; "
        "  text-align: left !important; "
        "}"
        "table th { "
        f"  background: {colors['card']} !important; "
        "  font-weight: 700 !important; "
        "}"
        "p, li, td, th, span, div { "
        f"  color: {colors['text']} !important; "
        "}"
        "b, strong { "
        f"  color: {colors['text']} !important; "
        "  font-weight: 700 !important; "
        "}"
        "</style></head><body>"
        f"{body_html}"
        "</body></html>"
    )


def _load_literature_file(filename: str) -> str:
    base_dir = Path(__file__).resolve().parent
    file_path = base_dir / "literature" / filename
    if not file_path.exists():
        return ""
    return file_path.read_text(encoding="utf-8")


def get_literature_html(model_name: str, is_dark: bool) -> str:
    filename = MODEL_LITERATURE_FILES.get(model_name)
    if not filename:
        body = (
            "<h2 style='margin-top:0;'>Literatura</h2>"
            "<p>Selecione um modelo para carregar a literatura.</p>"
        )
        return _wrap_html(body, is_dark)

    body_html = _load_literature_file(filename)
    if not body_html:
        body_html = (
            "<h2 style='margin-top:0;'>Literatura</h2>"
            "<p>Conteudo do modelo ainda nao foi adicionado.</p>"
        )
    return _wrap_html(body_html, is_dark)
