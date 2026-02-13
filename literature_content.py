from pathlib import Path


MODEL_LITERATURE_FILES = {
    "RandomForestClassifier": "random_forest/index.html",
    "RandomForestRegressor": "random_forest/index.html",
    "Logistic Regression": "logistic_regression/index.html",
    "Linear Regression": "linear_regression/index.html",
    "Ridge Regression": "ridge_regression/index.html",
    "SVR": "svr/index.html",
    "KNN": "knn/index.html",
    "Naive Bayes": "naive_bayes/index.html",
    "Gradient Boosting": "gradient_boosting/index.html",
    "Decision Tree": "decision_tree/index.html",
    "Decision Tree Regressor": "decision_tree/index.html",
    "Gradient Boosting Regressor": "gradient_boosting/index.html",
    "SVM": "svm/index.html",
    "XGBoost": "xgboost/index.html",
    "XGBoost Regressor": "xgboost/index.html",
}


def _wrap_html(body_html: str, is_dark: bool) -> str:
    colors = {
        "bg": "#121212" if is_dark else "#ffffff",
        "text": "#e6e6e6" if is_dark else "#1f1f1f",
        "muted": "#b5b5b5" if is_dark else "#5a5a5a",
        "accent": "#2e7d32" if is_dark else "#2e7d32",
        "card": "#1c1c1c" if is_dark else "#f7f4ef",
        "border": "#3a3a3a" if is_dark else "#d6d3cf",
    }
    return (
        "<html><body style=\""
        f"font-family: Segoe UI, Arial; color: {colors['text']}; background: {colors['bg']};"
        "\">"
        "<style>"
        "a { color: " + colors["accent"] + "; text-decoration: none; }"
        "code, pre { background: " + colors["card"] + "; border: 1px solid " + colors["border"] + "; }"
        "</style>"
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
