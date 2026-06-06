from flask import Flask, send_from_directory
from pathlib import Path
import markdown

app = Flask(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
BLOG_DIR = PROJECT_DIR / "blog"
POSTS_DIR = BLOG_DIR / "posts"


def render_page(title, body):
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1100px;
            margin: 40px auto;
            padding: 0 20px;
            line-height: 1.6;
            background: #f7f7f7;
            color: #222;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}
        th {{
            background: #f0f0f0;
        }}
        img {{
            max-width: 100%;
            height: auto;
        }}
        .nav {{
            margin-bottom: 20px;
        }}
        .nav a {{
            margin-right: 15px;
            text-decoration: none;
            font-weight: bold;
        }}
        .badge {{
            display: inline-block;
            background: #222;
            color: white;
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 13px;
            margin-bottom: 15px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="nav">
            <a href="/">Today</a>
            <a href="/history">History</a>
            <a href="/data/today.json">JSON</a>
        </div>
        <div class="badge">Technology Experiment • Not Financial Advice</div>
        {body}
    </div>
</body>
</html>
"""


@app.route("/")
def home():
    latest_file = BLOG_DIR / "latest.md"

    if not latest_file.exists():
        return render_page("FinMan AlphaCore", "<h1>No blog generated yet</h1>")

    html = markdown.markdown(
        latest_file.read_text(encoding="utf-8"),
        extensions=["tables"]
    )

    return render_page("FinMan AlphaCore Daily Picks", html)


@app.route("/history")
def history():
    posts = sorted(POSTS_DIR.glob("*.md"), reverse=True)

    links = "<h1>FinMan AlphaCore History</h1><ul>"
    for post in posts:
        date = post.stem
        links += f'<li><a href="/posts/{date}">{date}</a></li>'
    links += "</ul>"

    return render_page("FinMan AlphaCore History", links)


@app.route("/posts/<date>")
def post(date):
    post_file = POSTS_DIR / f"{date}.md"

    if not post_file.exists():
        return render_page("Post not found", "<h1>Post not found</h1>"), 404

    html = markdown.markdown(
        post_file.read_text(encoding="utf-8"),
        extensions=["tables"]
    )

    return render_page(f"FinMan AlphaCore {date}", html)


@app.route("/performance.svg")
def performance_graph():
    return send_from_directory(BLOG_DIR, "performance.svg")


@app.route("/data/<filename>")
def data_file(filename):
    return send_from_directory(BLOG_DIR / "data", filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
