import queue
import sys
import threading
import time
from collections import deque
from playwright.sync_api import sync_playwright

# ✅ now only imports this single line of logic
from .workspace import handle_message

USER_DATA_DIR = r"C:\Users\GeorgeC\AppData\Local\Google\Chrome\User Data\Profile 2"
SELF_CHAT_URL = "https://web.whatsapp.com/send/?phone=27796995695"
POLL_INTERVAL = 2  # seconds
SELF_NAME = "GeorgeC"

SELF_PREFIXES = ("noted", "task noted", "sorry", "⚠️", "error", "ok", "done")
SENT_CACHE = deque(maxlen=10)

LAST_OUTGOING = None  # tracks last message sent


# --------------------------------------------------------------------------
# 👇 Core utility methods (untouched)
# --------------------------------------------------------------------------

def is_from_self(text: str) -> bool:
    if not text:
        return False
    lowered = text.strip().lower()
    return any(lowered.startswith(p) for p in SELF_PREFIXES)


def find_input(page):
    strict_selector = 'div[contenteditable="true"][role="textbox"][aria-label^="Type"]'
    if page.query_selector(strict_selector):
        return strict_selector
    for sel in [
        'div[contenteditable="true"][data-tab="10"]',
        'div[aria-placeholder="Type a message"]',
        'footer div[contenteditable="true"]',
        'div._ak1l',
    ]:
        if page.query_selector(sel):
            return sel
    return None


def wait_for_input_box(page, timeout=60000):
    start = time.time()
    while time.time() - start < timeout / 1000:
        sel = find_input(page)
        if sel:
            return sel
        time.sleep(0.5)
    raise RuntimeError("Message input box not found.")


def is_just_sent(text: str) -> bool:
    if not text:
        return False
    return text.strip() in SENT_CACHE


def send_message(page, text):
    sel = find_input(page)
    if not sel:
        sel = wait_for_input_box(page, timeout=5000)
    if not sel:
        raise RuntimeError("Message input not found.")

    page.click(sel)

    lines = text.split("\n")
    for i, line in enumerate(lines):
        page.type(sel, line)
        if i != len(lines) - 1:
            page.keyboard.down("Shift")
            page.keyboard.press("Enter")
            page.keyboard.up("Shift")

    page.keyboard.press("Enter")

    SENT_CACHE.append(text.strip())
    global LAST_OUTGOING
    LAST_OUTGOING = text.strip()


def clean_quotes(text):
    text = text.strip()
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    return text


def command_reader(cmd_queue, stop_event):
    while not stop_event.is_set():
        try:
            cmd = input("> ")
        except (EOFError, KeyboardInterrupt):
            cmd_queue.put("exit")
            break
        cmd = cmd.strip()
        cmd_queue.put(cmd)
        if cmd.lower() == "exit":
            break


def read_last_message_text(page):
    js = """
    () => {
      const msgs = Array.from(document.querySelectorAll('div[role="row"] span.selectable-text'));
      if (!msgs.length) return null;
      return msgs[msgs.length - 1].innerText.trim();
    }
    """
    try:
        return page.evaluate(js)
    except Exception as e:
        print(f"[DEBUG] Error in read_last_message_text: {e}")
        return None


# --------------------------------------------------------------------------
# ✅ NEW: refactored message processor (lightweight)
# --------------------------------------------------------------------------

def process_incoming_message(page, message_text):
    cleaned = message_text.strip()
    if cleaned in ("```", "''", '""', "`", "'''"):
        print(f"[DEBUG] Ignoring noise message: {cleaned}")
        return

    lowered = cleaned.lower()
    blocked_starts = (
        "tyrone>", "*", "-", "•",
        "i cannot fulfill",
        "here are the stored birthdays"
    )
    if any(lowered.startswith(b) for b in blocked_starts):
        print(f"[DEBUG] Skipping blocked/self message: {message_text}")
        return

    if is_just_sent(cleaned):
        print(f"[DEBUG] Skipping echo of sent message: {cleaned}")
        return

    wh_starts = ("when ", "what ", "where ", "who ", "how ", "why ")
    if any(lowered.startswith(w) for w in wh_starts) and not cleaned.endswith("?"):
        cleaned = cleaned + "?"
        print(f"[DEBUG] Auto-appended '?': {cleaned}")

    print(f"[DEBUG] Processing incoming: {cleaned}")

    try:
        # 🔁 now routed through workspace
        reply = handle_message(cleaned)
        if reply:
            formatted = f"Tyrone> {reply}"
            print(f"[DEBUG] Sending reply: {formatted}")
            send_message(page, formatted)
    except Exception as e:
        print(f"Error while processing message: {e}")
        try:
            send_message(page, "⚠️ Error handling your message.")
        except:
            pass


# --------------------------------------------------------------------------
# 🚀 Main runner loop
# --------------------------------------------------------------------------

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            user_data_dir=USER_DATA_DIR,
            headless=False,
        )
        page = browser.new_page()
        page.goto(SELF_CHAT_URL)

        print("⏳ Loading WhatsApp...")

        try:
            wait_for_input_box(page)
            print("✅ WhatsApp ready and self-chat loaded.")
            print(">> You can now type 'send <message>' or 'exit' below. <<")
            last_seen_text = read_last_message_text(page)
            startup_boundary = last_seen_text
            print(f"[DEBUG] Startup boundary is: {startup_boundary}")
        except Exception as e:
            print("❌ Could not load WhatsApp:", e)
            browser.close()
            sys.exit(1)

        print("Listening for messages and commands...")
        print("Commands: send <message> | exit")

        cmd_queue = queue.Queue()
        stop_event = threading.Event()
        input_thread = threading.Thread(
            target=command_reader,
            args=(cmd_queue, stop_event),
            daemon=True,
        )
        input_thread.start()

        ready_for_messages = False

        try:
           while True:
                current_message = read_last_message_text(page)

                if current_message and current_message != last_seen_text:
                    if not ready_for_messages:
                        print(f"[DEBUG] Ignoring pre-launch message: {current_message}")
                        ready_for_messages = True
                        last_seen_text = current_message
                        continue

                    print(f"[DEBUG] New message detected: {current_message}")

                    # ✅ Ignore if it's part of the last outgoing message (multi-line echo protection)
                    if LAST_OUTGOING and current_message.strip() in LAST_OUTGOING:
                        print(f"[DEBUG] Ignoring echo (substring of last outgoing): {current_message}")
                        last_seen_text = current_message
                        continue

                    if not is_just_sent(current_message) and not is_from_self(current_message):
                        print(f"\n📩 INCOMING: {current_message}")
                        process_incoming_message(page, current_message)

                    last_seen_text = current_message


                # Commands
                try:
                    cmd = cmd_queue.get(timeout=POLL_INTERVAL)
                except queue.Empty:
                    continue

                if not cmd:
                    continue
                if cmd.lower() == "exit":
                    break
                if cmd.lower().startswith("send "):
                    text = clean_quotes(cmd[5:].strip())
                    if text:
                        try:
                            send_message(page, text)
                            print(f"✅ SENT: {text}")
                        except Exception as e:
                            print(f"❌ Failed to send: {e}")
                    else:
                        print("ℹ️ No message to send.")
                else:
                    print("Unrecognized command. Commands: send <message> | exit")

        finally:
            stop_event.set()
            if input_thread.is_alive():
                input_thread.join(timeout=1)
            try:
                browser.close()
            except Exception:
                print("⚠️ Browser was already closed or disconnected.")
            print("\n✅ Closed cleanly.")


if __name__ == "__main__":
    main()
