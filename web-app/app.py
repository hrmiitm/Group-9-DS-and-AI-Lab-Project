"""
webapp/app.py
Flask application factory. Self-contained — all code lives in webapp/.

Run from project root:
    python webapp/app.py

Or with flask CLI:
    FLASK_APP=webapp/app.py flask run
"""
import sys
import os

# Add webapp/ itself to sys.path so all internal packages resolve correctly.
# No external src/ dependency — everything is inside webapp/.
_WEBAPP_DIR = os.path.dirname(os.path.abspath(__file__))
if _WEBAPP_DIR not in sys.path:
    sys.path.insert(0, _WEBAPP_DIR)

from flask import Flask
from markupsafe import Markup

from config import SECRET_KEY, RESULTS_DIR, UPLOAD_DIR, TOOL_LABELS, TOOL_ICONS, simple_markdown


def create_app() -> Flask:
    # Use absolute template/static paths so app creation works regardless of import context.
    template_dir = os.path.join(_WEBAPP_DIR, "templates")
    static_dir = os.path.join(_WEBAPP_DIR, "static")
    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
    app.secret_key = SECRET_KEY

    # Ensure runtime directories exist
    RESULTS_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)

    # ── Jinja2 globals & filters ────────────────────────────────────────────
    app.jinja_env.globals.update(
        tool_labels=TOOL_LABELS,
        tool_icons=TOOL_ICONS,
    )

    @app.template_filter("markdown")
    def markdown_filter(text):
        return Markup(simple_markdown(text or ""))

    @app.template_filter("pretty_json")
    def pretty_json_filter(obj):
        import json
        return json.dumps(obj, indent=2, default=str)

    # ── Register blueprints ─────────────────────────────────────────────────
    from routes.main import main_bp
    from routes.api import api_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    return app


app = create_app()

if __name__ == "__main__":
    print("=" * 50)
    print("  Webetention — Fraud Job Detector")
    print("=" * 50)
    print(f"  Results : {RESULTS_DIR}")
    print(f"  Uploads : {UPLOAD_DIR}")
    print("  URL     : http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)
