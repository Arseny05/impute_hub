import os
from functools import wraps
import requests
from flask import Flask, render_template, request, session, redirect, url_for, Response, jsonify
from dotenv import load_dotenv
import json
import signal
import threading

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load config.json: {e}")
    return {}

CONFIG = load_config()

load_dotenv()

PORT = int(os.getenv("PORT", CONFIG.get("service", {}).get("port", 8000)))
HOST = os.getenv("HOST", CONFIG.get("service", {}).get("host", "0.0.0.0"))

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "super-secret-default-key")
PORTAL_PASSWORD = os.getenv("PORTAL_PASSWORD", "admin420")

CONFIG_SERVICES = CONFIG.get("services", {})
SERVICES = {
    "collector": os.getenv("COLLECTOR_URL", CONFIG_SERVICES.get("collector", "http://collector_service:8501")),
    "impute": os.getenv("IMPUTE_URL", CONFIG_SERVICES.get("impute", "http://impute_service:8501")),
    "visualization": os.getenv("VISUALIZATION_URL", CONFIG_SERVICES.get("visualization", "http://visualization_service:8501")),
    "storage": os.getenv("STORAGE_URL", CONFIG_SERVICES.get("storage", "http://storage_service:5001")),
    "assistant": os.getenv("ASSISTANT_URL", CONFIG_SERVICES.get("assistant", "http://localhost:8502")), 
}

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/stop_system', methods=['POST'])
def stop_system():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    session.clear()

    def shutdown():
        os._exit(0)

    threading.Timer(0.3, shutdown).start()
    return redirect(url_for('login'))


def login_required(f):

    @wraps(f)
    def decoration_function(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decoration_function

@app.route("/login", methods=['POST', 'GET'])
def login():
    error = None
    if request.method == "POST":
        password = request.form.get("password")
        if password == PORTAL_PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        else:
            error = "Invalid password. Please try again."
    
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    return redirect(url_for("login"))

@app.route("/")
@login_required
def dashboard():
    return render_template("dashboard.html", services=SERVICES)

@app.route("/api/<service_name>/<path:endpoint>", methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy_api(service_name, endpoint):
    if service_name not in SERVICES:
        return {"error": "Service not found"}, 404

    
    target_url = f"{SERVICES[service_name]}/api/{service_name}/{endpoint}"
    
    try:
        resp = requests.request(
            method=request.method,
            url=target_url,
            params=request.args,  
            headers={key: value for (key, value) in request.headers if key != 'Host'},
            data=request.get_data(),
            cookies=request.cookies,
            allow_redirects=False,
            timeout=10
        )
        return Response(
            resp.content,
            resp.status_code,
            {key: value for key, value in resp.headers.items() if key.lower() not in ['content-encoding', 'transfer-encoding']}
        )
    except requests.exceptions.RequestException as e:
        return {"error": f"Gateway timeout or connection error: {str(e)}"}, 502

if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False, threaded=True)