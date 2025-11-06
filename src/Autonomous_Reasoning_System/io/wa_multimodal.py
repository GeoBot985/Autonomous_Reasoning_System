import os
import time
import base64
from pathlib import Path
from playwright.sync_api import sync_playwright

# ✅ Config
USER_DATA_DIR = r"C:\Users\GeorgeC\AppData\Local\Google\Chrome\User Data\Profile 2"
SELF_CHAT_URL = "https://web.whatsapp.com/send/?phone=27796995695"
POLL_INTERVAL = 3  # seconds
SAVE_DIR = Path("data/incoming_media")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# IMAGE CAPTURE (full version)
# --------------------------------------------------------------------------
def capture_full_image(page, filename):
    """
    Opens the latest thumbnail image in the chat, screenshots the full-size
    preview, then closes the viewer.
    """
    try:
        # 1️⃣ Locate the most recent image thumbnail
        js_thumb = """
        () => {
          const imgs = Array.from(document.querySelectorAll('img[src^="blob:"]'));
          return imgs.length ? imgs[imgs.length - 1] : null;
        }
        """
        thumb = page.evaluate_handle(js_thumb)
        if not thumb:
            print("[DEBUG] No thumbnail found.")
            return None

        # 2️⃣ Click to open WhatsApp’s full-image viewer
        thumb.click()
        page.wait_for_selector('img[src^="blob:"]', timeout=5000)

        # 3️⃣ Grab the (now larger) image
        large_img = page.query_selector('img[src^="blob:"]')
        path = SAVE_DIR / filename
        if large_img:
            large_img.screenshot(path=str(path))
            print(f"[DEBUG] Captured full image → {path}")
        else:
            thumb.screenshot(path=str(path))
            print(f"[DEBUG] Fallback: thumbnail screenshot → {path}")

        # 4️⃣ Close the viewer
        page.keyboard.press("Escape")
        thumb.dispose()
        return str(path)

    except Exception as e:
        print(f"[DEBUG] capture_full_image error: {e}")
        return None

# --------------------------------------------------------------------------
# SOUND CAPTURE (full version)
# --------------------------------------------------------------------------
def save_voice_note(page, blob_url):
    """
    Reads a WhatsApp audio blob (voice note) via FileReader inside the page
    context and writes it to data/incoming_media as .ogg.
    """
    try:
        js = f"""
        async () => {{
          try {{
            const blob = await fetch("{blob_url}").then(r => r.blob());
            return await new Promise((resolve, reject) => {{
              const reader = new FileReader();
              reader.onloadend = () => resolve(reader.result.split(',')[1]);
              reader.onerror = reject;
              reader.readAsDataURL(blob);
            }});
          }} catch (e) {{
            return null;
          }}
        }}
        """
        b64 = page.evaluate(js)
        if not b64:
            print("[DEBUG] No base64 returned for voice note (fetch failed).")
            return None

        filename = f"wa_{int(time.time())}.ogg"
        path = SAVE_DIR / filename
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64))
        return str(path)

    except Exception as e:
        print(f"[DEBUG] save_voice_note error: {e}")
        return None


# --------------------------------------------------------------------------
# UNIFIED MESSAGE READER (text, image, voice)
# --------------------------------------------------------------------------
def read_last_message(page):
    js = """
    () => {
      const rows = Array.from(document.querySelectorAll('div[role="row"]'));
      if (!rows.length) return null;
      const last = rows[rows.length - 1];

      // --- Voice note detection ---
      const voiceBubble = last.querySelector('div[aria-label*="audio"], div[aria-label*="Voice"], div[data-testid="audio-playback"]');
      const playBtn = last.querySelector('button[aria-label*="Play"]');
      const durationSpan = last.querySelector('span[aria-label*="second"], span[aria-label*="minute"]');
      const text = last.querySelector('span.selectable-text')?.innerText?.trim() || null;

      if (voiceBubble || playBtn) {
        const duration = durationSpan ? durationSpan.getAttribute("aria-label") : null;
        // voice messages often hide the real blob URL in a data attribute
        const blob = last.querySelector('audio[src^="blob:"], div[role="button"][data-plain-text="true"], div[role="button"][src^="blob:"]');
        const src = blob ? (blob.getAttribute("src") || null) : null;
        return { type: 'voice', src, duration, text };
      }

      // --- Image ---
      const img = last.querySelector('img[src^="blob:"]');
      const caption = last.querySelector('span.selectable-text')?.innerText?.trim() || null;
      if (img) return { type: 'image', src: img.getAttribute('src'), caption };

      // --- Text ---
      if (text) return { type: 'text', text };

      return null;
    }
    """
    try:
        return page.evaluate(js)
    except Exception as e:
        print(f"[DEBUG] JS error: {e}")
        return None

# --------------------------------------------------------------------------
# SOUND CAPTURE (full version)
# --------------------------------------------------------------------------
def capture_voice_note(page, context):
    """
    Clicks play on the most recent voice message and intercepts the
    network response that contains the actual audio data.
    Saves it as .ogg in data/incoming_media/.
    """
    saved_file = None

    def handle_response(response):
        nonlocal saved_file
        try:
            # Only capture audio responses
            if "audio" in (response.request.resource_type or "").lower():
                content_type = response.headers.get("content-type", "")
                if content_type.startswith("audio") and not saved_file:
                    data = response.body()
                    ext = ".ogg" if "ogg" in content_type else ".opus"
                    path = SAVE_DIR / f"wa_{int(time.time())}{ext}"
                    with open(path, "wb") as f:
                        f.write(data)
                    saved_file = str(path)
                    print(f"[DEBUG] Intercepted and saved audio → {path}")
        except Exception as e:
            print(f"[DEBUG] handle_response error: {e}")

    # Listen for network responses temporarily
    context.on("response", handle_response)

    try:
        # Find and click the last Play button
        js_play = """
        () => {
          const buttons = Array.from(document.querySelectorAll('button[aria-label*="Play"], button[data-testid="audio-play"]'));
          if (buttons.length) {
            const last = buttons[buttons.length - 1];
            last.click();
            return true;
          }
          return false;
        }
        """
        played = page.evaluate(js_play)
        if not played:
            print("[DEBUG] No Play button found.")
            return None

        # Wait briefly to allow network to load audio
        page.wait_for_timeout(4000)

    except Exception as e:
        print(f"[DEBUG] capture_voice_note error: {e}")

    finally:
        # Stop listening
        context.remove_listener("response", handle_response)

    return saved_file


# --------------------------------------------------------------------------
# MAIN LOOP
# --------------------------------------------------------------------------
def main():
    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            user_data_dir=USER_DATA_DIR, headless=False
        )
        page = browser.new_page()
        page.goto(SELF_CHAT_URL)
        print("⏳ Loading WhatsApp...")

        time.sleep(10)
        print("✅ Ready — watching for text, image, and voice messages.")

        last_seen = None
        while True:
            msg = read_last_message(page)
            if not msg:
                time.sleep(POLL_INTERVAL)
                continue

            if msg != last_seen:
                last_seen = msg

                # 📝 TEXT
                if msg["type"] == "text":
                    print(f"\n📩 TEXT: {msg['text']}")

                # 🖼️ IMAGE
                elif msg["type"] == "image":
                    print("\n📩 IMAGE RECEIVED")
                    filename = f"wa_{int(time.time())}.jpg"
                    file_path = capture_full_image(page, filename)
                    print(f"📎 Saved: {file_path}")
                    if msg.get("caption"):
                        print(f"💬 Caption: {msg['caption']}")

                # 🎙️ VOICE
                elif msg["type"] == "voice":
                    print("\n🎙️ VOICE NOTE DETECTED")
                    print(f"🔗 Blob: {msg['src']}")
                    file_path = capture_voice_note(page, browser)
                    print(f"🎧 Saved: {file_path}")
                    if msg.get("duration"):
                        print(f"⏱ Duration: {msg['duration']}")
                    if msg.get("text"):
                        print(f"💬 Caption: {msg['text']}")


            time.sleep(POLL_INTERVAL)



if __name__ == "__main__":
    main()
