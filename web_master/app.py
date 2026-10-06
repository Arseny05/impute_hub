import os
from functools import wraps
import requests
from flask import Flask, render_template, request, session, redirect, url_for, Response
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "super-secret-default-key")
PORTAL_PASSWORD = os.getenv("PORTAL_PASSWORD", "admin420")

SERVICES = {
    "collector": os.getenv("COLLECTOR_URL", "http://localhost:5002"),
    "impute": os.getenv("IMPUTE_URL", "http://localhost:5004"),
    "visualization": os.getenv("VISUALIZATION_URL", "http://localhost:5005"),
    "storage": os.getenv("STORAGE_URL", "http://localhost:5001"),
}

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
            headers={key: value for (key, value) in request.headers if key != 'Host'},
            data=request.get_data(),
            cookies=request.cookies,
            allow_redirects=False,
            timeout=10
        )

        return Response(resp.content,
                        resp.status_code,
                        {key: value for key, value in resp.headers.items() if key.lower() not in ['content-encoding', 'transfer-encoding']})
    except requests.exceptions.RequestException as e:
        return {"error": f"Gateway timeout or connection error: {str(e)}"}, 502

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("WEB_MASTER_PORT", 8000)), debug=True)