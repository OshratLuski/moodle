#!/usr/bin/env python3
import os, json, re, argparse
import requests
import boto3

GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
REPO = os.environ["REPO"]
PR_NUMBER = os.environ["PR_NUMBER"]
AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")
MODEL_ID = os.environ.get("ANTHROPIC_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "4000"))

GH_API = "https://api.github.com"

# Debug toggle (set DEBUG=1 in workflow env to enable)
DEBUG = os.environ.get("DEBUG", "0") == "1"
def debug(msg: str):
    if DEBUG:
        print(f"DEBUG: {msg}")

# ---------- GitHub helpers ----------
def gh_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

def get_changed_files():
    url = f"{GH_API}/repos/{REPO}/pulls/{PR_NUMBER}/files"
    files = []
    page = 1
    while True:
        resp = requests.get(url, headers=gh_headers(), params={"page": page, "per_page": 100})
        resp.raise_for_status()
        chunk = resp.json()
        if not chunk:
            break
        files.extend(chunk)
        page += 1
    return files

def get_full_pr_diff_text() -> str:
    """
    Fallback: fetch entire PR diff as unified text using 'diff' media type.
    """
    url = f"{GH_API}/repos/{REPO}/pulls/{PR_NUMBER}"
    headers = gh_headers().copy()
    headers["Accept"] = "application/vnd.github.v3.diff"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.text or ""

def create_or_update_comment(body: str):
    list_url = f"{GH_API}/repos/{REPO}/issues/{PR_NUMBER}/comments"
    resp = requests.get(list_url, headers=gh_headers())
    resp.raise_for_status()
    comments = resp.json()
    marker = "<!-- ai-code-review:bedrock-claude -->"

    existing = next((c for c in comments if c.get("body","").startswith(marker)), None)
    payload = {"body": f"{marker}\n{body}"}

    if existing:
        edit_url = f"{GH_API}/repos/{REPO}/issues/comments/{existing['id']}"
        r = requests.patch(edit_url, headers=gh_headers(), json=payload)
        r.raise_for_status()
    else:
        r = requests.post(list_url, headers=gh_headers(), json=payload)
        r.raise_for_status()

# ---------- Bedrock (Anthropic Messages) ----------
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

SYSTEM_PROMPT = """You are a senior Moodle developer doing inline code review on a GitHub Pull Request.
Return ONLY JSON (no markdown) with this schema:
{
  "comments": [
    {
      "path": "relative/file/path.php",
      "anchor_text": "exact line as it appears ADDED in the diff (without the leading '+')",
      "message": "short precise review comment, Moodle-specific if relevant",
      "suggestion": "optional: full replacement text for a GitHub suggestion block"
    }
  ]
}
Rules:
- Comment ONLY on lines that were ADDED/CHANGED in this PR (use the diff).
- Use 'anchor_text' to identify the added line to attach to.
- Up to 20 comments. Be concrete and actionable. Prefer one comment per issue.
- If you propose a fix, fill 'suggestion' with the exact replacement content (no backticks).
- Focus on Moodle standards (PSR-12, frankenstyle), security (XSS/CSRF/SQLi), API usage, I18N, accessibility, docs, tests, performance.
"""

USER_PREFIX = """Review PR {pr} in {repo}. Here is a unified diff chunk.
Output ONLY the JSON described above. Do not write anything else.

Diff:

{diff}

"""

def parse_comments_json(text: str) -> list[dict]:
    # Defensive parse: try to locate the first {...} JSON object in the text
    try:
        # common: model returns pure JSON
        obj = json.loads(text)
        return obj.get("comments", []) if isinstance(obj, dict) else []
    except Exception:
        # try to extract JSON substring
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(text[start:end+1])
                return obj.get("comments", []) if isinstance(obj, dict) else []
            except Exception:
                return []
        return []
    
def patch_line_positions(patch: str) -> dict:
    """
    Return a mapping from patch line index (1-based 'position') to the actual text line (with +/-/space).
    """
    lines = patch.splitlines()
    positions = {}
    pos = 1
    for ln in lines:
        # All lines in 'patch' contribute to position counting (GitHub counts within the patch)
        positions[pos] = ln
        pos += 1
    return positions

def find_position_for_anchor(patch: str, anchor_text: str) -> int | None:
    """
    Find the first position in patch where an added line '+...' matches the given anchor_text.
    anchor_text must match the content after the '+' exactly.
    """
    positions = patch_line_positions(patch)
    target = "+" + anchor_text.strip("\n")
    for pos, ln in positions.items():
        if ln == target:
            return pos
    return None

def post_inline_review(comments: list[dict]):
    """
    POST one review with multiple inline comments.
    Each item must have: path, position, body
    """
    url = f"{GH_API}/repos/{REPO}/pulls/{PR_NUMBER}/reviews"
    pr = requests.get(f"{GH_API}/repos/{REPO}/pulls/{PR_NUMBER}", headers=gh_headers(), timeout=30)
    pr.raise_for_status()
    commit_id = pr.json()["head"]["sha"]
    payload = {"event": "COMMENT", "commit_id": commit_id, "comments": comments}
    r = requests.post(url, headers=gh_headers(), json=payload, timeout=30)
    r.raise_for_status()

def build_unified_diff(files):
    """
    Build a unified diff from the GitHub 'files' API.
    If files lack 'patch' (binary/large), returns empty string and caller may fallback.
    """
    diffs = []
    for f in files:
        filename = f.get("filename")
        status = f.get("status")
        has_patch = "patch" in f

        debug(f"file: {filename} status={status} has_patch={has_patch}")
        if not has_patch:
            continue

        if status == "removed":
            header = f"--- a/{filename}\n+++ /dev/null\n"
        else:
            header = f"--- a/{filename}\n+++ b/{filename}\n"
        patch = f["patch"]
        diffs.append(header + patch)
    return "\n".join(diffs)

def call_bedrock(prompt: str) -> str:
    req = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": [{"type":"text", "text": prompt}]}
        ]
    }
    resp = bedrock.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(req),
    )
    body = json.loads(resp["body"].read())
    parts = body.get("content", [])
    text = ""
    for p in parts:
        if p.get("type") == "text":
            text += p.get("text","")
    return text.strip()

def extract_verdict(markdown: str) -> str:
    """
    Naively infer a verdict string from the combined markdown:
    approve | comment | request_changes
    """
    m = re.search(r'Overall verdict.*?:\s*(.+)', markdown, re.IGNORECASE | re.DOTALL)
    text = (m.group(1) if m else markdown).lower()
    if "request changes" in text or "changes requested" in text:
        return "request_changes"
    if "approve" in text or "lgtm" in text:
        return "approve"
    return "comment"

def chunk_text(text: str, max_chars: int = 12000):
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        split = text.rfind("\n@@", start, end)
        if split == -1 or split <= start + 1000:
            split = text.rfind("\n", start, end)
            if split == -1 or split <= start:
                split = end
        chunks.append(text[start:split])
        start = split
    return chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit-verdict", action="store_true")
    args = parser.parse_args()

    # 1) קבצים ששונו
    files = get_changed_files()
    debug(f"files count from GitHub API: {len(files)}")

    # אזהרת theme/ למעט theme/petel
    theme_warnings = []
    for f in files:
        filename = f.get("filename", "")
        if filename.startswith("theme/") and not filename.startswith("theme/petel"):
            theme_warnings.append(f"- {filename}")
    if theme_warnings:
        warning_text = (
            "⚠️ **Notice:** Changes detected in theme directories outside of `theme/petel`.\n\n"
            "The following files were modified:\n"
            + "\n".join(theme_warnings) +
            "\n\nPlease avoid editing themes other than `theme/petel`."
        )
        create_or_update_comment(warning_text)

    if not files:
        create_or_update_comment("No changed files detected.")
        if args.emit_verdict:
            print("comment", end="")
        return

    # נשתמש ב-patch מה-files API כדי לחשב positions, וגם נבנה unified לצורך הקשר למודל
    patch_by_path = {f["filename"]: f.get("patch", "") for f in files if f.get("patch")}
    unified = build_unified_diff(files)
    debug(f"unified diff length (from files API): {len(unified)}")

    # אם אין unified (למשל בינארי/גדול), נביא דיפ מלא להקשר למודל (positions עדיין מה-files API)
    if not unified:
        debug("no patches via files API; fetching full PR diff text…")
        unified = get_full_pr_diff_text()
        debug(f"unified diff length (full PR diff): {len(unified)}")

    if not unified and not patch_by_path:
        create_or_update_comment("Changed files are binary or too large; no textual diff available.")
        if args.emit_verdict:
            print("comment", end="")
        return

    # 2) שולחים למודל בקבצים (chunks) ומקבלים JSON של הערות אינליין
    chunks = chunk_text(unified if unified else "")
    raw_comments = []
    for i, chunk in enumerate(chunks, 1):
        user = USER_PREFIX.format(repo=REPO, pr=PR_NUMBER, diff=chunk)
        debug(f"sending chunk {i}/{len(chunks)} to model; chunk_len={len(chunk)}")
        out = call_bedrock(user)
        raw_comments.extend(parse_comments_json(out))

    # 3) מעבדים JSON -> הערות Review עם position
    review_comments = []
    for c in raw_comments:
        path = c.get("path")
        anchor = (c.get("anchor_text") or "").rstrip("\n")
        msg = (c.get("message") or "").strip()
        suggestion = c.get("suggestion")

        if not path or not anchor or not msg:
            continue
        patch = patch_by_path.get(path)
        if not patch:
            # אם אין לנו patch לאותו קובץ (למשל בינארי/גדול) – אין איך למפות position
            continue

        pos = find_position_for_anchor(patch, anchor)
        if pos is None:
            # אם לא מצאנו התאמה לשורת '+', מדלגים בשקט
            debug(f"no position match for path={path} anchor={anchor[:80]}")
            continue

        body = msg
        if suggestion:
            body += "\n\n```suggestion\n" + suggestion.rstrip("\n") + "\n```"

        review_comments.append({"path": path, "position": pos, "body": body})

    # 4) פרסום: רק הערות אינליין; ללא פוסט סיכום
    if review_comments:
        post_inline_review(review_comments)
    else:
        # פידבק מינימלי כדי שתדעי שהריצה קרתה
        create_or_update_comment("AI review: no actionable inline comments on changed lines.")

    # 5) פלט ל-gate (אם משתמשים בו בעתיד) – נשאר 'comment' תמיד
    if args.emit_verdict:
        print("comment", end="")

if __name__ == "__main__":
    main()
