#!/usr/bin/env python3
"""Extract successful calibration trip names from a Feishu group.

Usage modes:
  A) OAuth (external groups): python extract_feishu_trips.py --oauth --chat-id oc_xxx
     Opens browser for user login -> reads messages with user's permission.

  B) Bot in group: python extract_feishu_trips.py [--chat-id oc_xxx]
     Requires bot to be added to the group.

  C) Offline: python extract_feishu_trips.py --from-text /path/to/messages.txt
     Parses trip names from a local text file.
"""
import argparse
import http.server
import json
import os
import re
import sys
import threading
import time
import urllib.request
import urllib.error
import webbrowser
from datetime import datetime, timedelta

APP_ID = "cli_a94086e7e77a5bd3"
APP_SECRET = "tiHWU1KFB5Gu5jFpkQ2htcyIicKDf7qb"
BASE_URL = "https://open.feishu.cn/open-apis"
TRIP_PATTERN = re.compile(r"YR-[A-Za-z0-9]+-\d+_\d{8}_\d{6}")
SUCCESS_KEYWORDS = ["标定成功", "calibration.*success", "PASS", "pass", "成功"]


def _api_request(method, path, token=None, body=None, params=None):
    url = BASE_URL + path
    if params:
        url += "?" + "&".join("{}={}".format(k, v) for k, v in params.items())
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json; charset=utf-8")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode() if e.fp else ""
        print("[feishu] HTTP {} for {}: {}".format(e.code, path, body_text[:300]))
        return {"code": e.code, "msg": body_text[:300]}


def get_tenant_token():
    resp = _api_request("POST", "/auth/v3/tenant_access_token/internal",
                        body={"app_id": APP_ID, "app_secret": APP_SECRET})
    if resp.get("code") != 0:
        raise RuntimeError("Failed to get token: {}".format(resp))
    return resp["tenant_access_token"]


OAUTH_REDIRECT_PORT = 19876
OAUTH_REDIRECT_URI = "http://localhost:{}".format(OAUTH_REDIRECT_PORT)
_oauth_code_holder = {"code": None}


class _OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        qs = parse_qs(urlparse(self.path).query)
        code = qs.get("code", [None])[0]
        if code:
            _oauth_code_holder["code"] = code
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                "Authorization successful. You can close this tab.".encode())
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing code parameter.")

    def log_message(self, fmt, *args):
        pass


def get_user_access_token_via_oauth():
    """Launch OAuth flow: open browser -> user logs in -> capture code -> get user token."""
    auth_url = (
        "https://open.feishu.cn/open-apis/authen/v1/authorize"
        "?app_id={}&redirect_uri={}&scope=im:message im:chat"
    ).format(APP_ID, OAUTH_REDIRECT_URI)

    server = http.server.HTTPServer(("127.0.0.1", OAUTH_REDIRECT_PORT), _OAuthCallbackHandler)
    server.timeout = 120

    print("[oauth] Starting local callback server on port {}...".format(OAUTH_REDIRECT_PORT))
    print("[oauth] Please open this URL in your browser to authorize:")
    print()
    print("  " + auth_url)
    print()
    print("[oauth] Or copy-paste the URL above into a browser where you are logged into Feishu")

    try:
        webbrowser.open(auth_url)
    except Exception:
        pass

    print("[oauth] Waiting for authorization callback (timeout 120s)...")
    while _oauth_code_holder["code"] is None:
        server.handle_request()

    code = _oauth_code_holder["code"]
    print("[oauth] Got authorization code, exchanging for token...")

    resp = _api_request("POST", "/authen/v1/oidc/access_token",
                        token=get_app_access_token(),
                        body={"grant_type": "authorization_code", "code": code})
    if resp.get("code") != 0:
        alt = _api_request("POST", "/authen/v1/access_token",
                           token=get_app_access_token(),
                           body={"grant_type": "authorization_code", "code": code})
        if alt.get("code") != 0:
            raise RuntimeError("Token exchange failed: {}".format(alt))
        return alt.get("data", {}).get("access_token")

    return resp.get("data", {}).get("access_token")


def get_app_access_token():
    """Get app_access_token (used in OAuth token exchange)."""
    resp = _api_request("POST", "/auth/v3/app_access_token/internal",
                        body={"app_id": APP_ID, "app_secret": APP_SECRET})
    if resp.get("code") != 0:
        raise RuntimeError("Failed to get app token: {}".format(resp))
    return resp.get("app_access_token")


def _search_public_chats(token, keyword):
    """Search for public/visible chats by keyword (requires search:chat scope)."""
    params = {"query": keyword, "page_size": "20"}
    resp = _api_request("GET", "/im/v1/chats/search", token=token, params=params)
    if resp.get("code") != 0:
        print("[feishu] Search chats: {} (may need search:chat scope)".format(
            resp.get("msg", resp.get("code"))))
        return []
    return resp.get("data", {}).get("items", [])


def find_target_chat(token, keyword="calibration"):
    """Find the chat matching the keyword. Try bot membership first, then public search."""
    page_token = ""
    all_chats = []
    while True:
        params = {"page_size": "50"}
        if page_token:
            params["page_token"] = page_token
        resp = _api_request("GET", "/im/v1/chats", token=token, params=params)
        if resp.get("code") != 0:
            print("[feishu] List chats failed: {}".format(resp.get("msg")))
            break
        items = resp.get("data", {}).get("items", [])
        all_chats.extend(items)
        if not resp["data"].get("has_more"):
            break
        page_token = resp["data"].get("page_token", "")

    print("[feishu] Bot is in {} chats".format(len(all_chats)))
    for c in all_chats:
        name = c.get("name", "")
        print("  - {} (chat_id={})".format(name, c.get("chat_id", "")))
        if keyword.lower() in name.lower():
            return c["chat_id"], name

    if not all_chats:
        print("[feishu] Bot not in any group, trying public search...")
        search_results = _search_public_chats(token, keyword)
        if search_results:
            for c in search_results:
                name = c.get("name", "")
                chat_id = c.get("chat_id", "")
                print("  [search] {} (chat_id={})".format(name, chat_id))
                if keyword.lower() in name.lower():
                    print("[feishu] Found via search! Attempting to join...")
                    join_resp = _api_request(
                        "PATCH", "/im/v1/chats/{}/members/me_join".format(chat_id),
                        token=token)
                    if join_resp.get("code") == 0:
                        print("[feishu] Successfully joined group: {}".format(name))
                        return chat_id, name
                    else:
                        print("[feishu] Auto-join failed: {} — manual add required".format(
                            join_resp.get("msg", join_resp.get("code"))))
                        print("[feishu] Use this chat_id directly: --chat-id {}".format(chat_id))
                        return chat_id, name

    return None, None


def fetch_messages(token, chat_id, start_ts, end_ts):
    """Fetch all messages in a chat within the time range."""
    messages = []
    page_token = ""
    start_ms = str(int(start_ts * 1000))
    end_ms = str(int(end_ts * 1000))
    while True:
        params = {
            "container_id_type": "chat",
            "container_id": chat_id,
            "start_time": start_ms,
            "end_time": end_ms,
            "sort_type": "ByCreateTimeDesc",
            "page_size": "50",
        }
        if page_token:
            params["page_token"] = page_token
        resp = _api_request("GET", "/im/v1/messages", token=token, params=params)
        if resp.get("code") != 0:
            print("[feishu] Fetch messages failed: {}".format(resp.get("msg")))
            break
        items = resp.get("data", {}).get("items", [])
        messages.extend(items)
        print("[feishu] Fetched {} messages (total so far: {})".format(len(items), len(messages)))
        if not resp["data"].get("has_more"):
            break
        page_token = resp["data"].get("page_token", "")
        time.sleep(0.2)
    return messages


def extract_trips_from_messages(messages):
    """Extract unique trip names from messages that indicate success."""
    success_re = re.compile("|".join(SUCCESS_KEYWORDS), re.IGNORECASE)
    trips = set()
    for msg in messages:
        msg_type = msg.get("msg_type", "")
        body_str = msg.get("body", {}).get("content", "")
        if not body_str:
            continue

        try:
            content = json.loads(body_str)
            text = content.get("text", "")
        except (json.JSONDecodeError, AttributeError):
            text = body_str

        if msg_type == "post":
            try:
                post = json.loads(body_str)
                parts = []
                for lang_content in post.values():
                    if isinstance(lang_content, dict):
                        title = lang_content.get("title", "")
                        if title:
                            parts.append(title)
                        for block in lang_content.get("content", []):
                            for elem in block:
                                if elem.get("tag") == "text":
                                    parts.append(elem.get("text", ""))
                text = " ".join(parts)
            except (json.JSONDecodeError, AttributeError):
                pass

        found_trips = TRIP_PATTERN.findall(text)
        if not found_trips:
            continue

        is_success = bool(success_re.search(text))
        if is_success or found_trips:
            for t in found_trips:
                trips.add(t)

    return sorted(trips)


def extract_from_text_file(text_path, output_path):
    """Offline mode: extract trip names from a plain text file (e.g. copied messages)."""
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    all_trips = TRIP_PATTERN.findall(text)
    success_re = re.compile("|".join(SUCCESS_KEYWORDS), re.IGNORECASE)
    lines = text.split("\n")

    trips = set()
    for line in lines:
        found = TRIP_PATTERN.findall(line)
        if found:
            trips.update(found)

    trips = sorted(trips)
    print("[offline] Found {} unique trip names from text file".format(len(trips)))
    if trips:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            for t in trips:
                f.write(t + "\n")
        print("[offline] Saved to {}".format(output_path))
        for t in trips:
            print("  " + t)
    else:
        print("[offline] No YR-XXX-N_YYYYMMDD_HHMMSS patterns found in text file.")
    return trips


def main():
    parser = argparse.ArgumentParser(
        description="Extract calibration trip names from Feishu group or local text file")
    parser.add_argument("--output",
                        default="/mnt/drtraining/user/dahailu/data/bevcalib/bag_lists/feishu_calib_trips.txt")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--chat-id", default=None,
                        help="Directly specify chat_id (e.g. oc_xxx). Find it in Feishu Web URL.")
    parser.add_argument("--keyword", default="calibration",
                        help="Group name keyword to search")
    parser.add_argument("--from-text", default=None,
                        help="Offline mode: extract trips from a local text file")
    parser.add_argument("--oauth", action="store_true",
                        help="Use OAuth user authorization to access external groups. "
                             "Opens browser for login, captures user_access_token.")
    parser.add_argument("--user-token", default=None,
                        help="Directly provide a user_access_token obtained from Feishu API Explorer. "
                             "Get one at: https://open.feishu.cn/api-explorer/ -> Authorization -> user_access_token")
    args = parser.parse_args()

    if args.from_text:
        print("=== Feishu Trip Extractor (Offline Mode) ===")
        extract_from_text_file(args.from_text, args.output)
        return

    print("=== Feishu Trip Extractor ===")
    print("Looking for trips from the last {} days".format(args.days))

    if args.user_token:
        print("[feishu] Using provided user_access_token for external group access")
        if not args.chat_id:
            print("ERROR: --user-token requires --chat-id oc_xxx")
            sys.exit(1)
        token = args.user_token
        chat_id = args.chat_id
    elif args.oauth:
        print("[feishu] OAuth mode: will use user_access_token for external group access")
        if not args.chat_id:
            print("ERROR: --oauth requires --chat-id oc_xxx")
            sys.exit(1)
        token = get_user_access_token_via_oauth()
        print("[feishu] User access token acquired!")
        chat_id = args.chat_id
    else:
        token = get_tenant_token()
        print("[feishu] Tenant token acquired")

        if args.chat_id:
            chat_id = args.chat_id
            print("[feishu] Using provided chat_id: {}".format(chat_id))
        else:
            chat_id, chat_name = find_target_chat(token, args.keyword)
            if not chat_id:
                print("\n" + "=" * 60)
                print("ERROR: Could not find the target group via API.")
                print()
                print("=== Method 1: OAuth for external groups (RECOMMENDED) ===")
                print("  python extract_feishu_trips.py --oauth --chat-id oc_xxx")
                print()
                print("=== Method 2: Offline mode (paste messages) ===")
                print("  python extract_feishu_trips.py --from-text /path/to/messages.txt")
                print("=" * 60)
                sys.exit(1)
            print("[feishu] Found group: {} (chat_id={})".format(chat_name, chat_id))

    now = datetime.now()
    start = now - timedelta(days=args.days)
    start_ts = start.timestamp()
    end_ts = now.timestamp()
    print("[feishu] Time range: {} ~ {}".format(start.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")))

    messages = fetch_messages(token, chat_id, start_ts, end_ts)
    print("[feishu] Total messages fetched: {}".format(len(messages)))

    trips = extract_trips_from_messages(messages)
    print("[feishu] Found {} unique trip names".format(len(trips)))

    if trips:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            for t in trips:
                f.write(t + "\n")
        print("[feishu] Saved to {}".format(args.output))
        print("\nTrip names:")
        for t in trips:
            print("  " + t)
    else:
        print("[feishu] No trips found. Try --from-text for offline extraction.")


if __name__ == "__main__":
    main()
