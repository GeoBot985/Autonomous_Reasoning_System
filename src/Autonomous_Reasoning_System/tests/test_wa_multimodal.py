import pytest
from unittest.mock import MagicMock, patch, mock_open
import base64
from Autonomous_Reasoning_System.io.wa_multimodal import (
    read_last_message,
    capture_full_image,
    save_voice_note,
    capture_voice_note
)

@pytest.fixture
def mock_page():
    return MagicMock()

def test_read_last_message_text(mock_page):
    # Setup: JS evaluation returns a text message object
    mock_page.evaluate.return_value = {'type': 'text', 'text': 'Hello world'}
    result = read_last_message(mock_page)
    assert result['type'] == 'text'
    assert result['text'] == 'Hello world'

def test_read_last_message_image(mock_page):
    # Setup: JS evaluation returns an image message object
    mock_page.evaluate.return_value = {'type': 'image', 'src': 'blob:http://...', 'caption': 'Look at this'}
    result = read_last_message(mock_page)
    assert result['type'] == 'image'
    assert result['src'] == 'blob:http://...'
    assert result['caption'] == 'Look at this'

def test_read_last_message_voice(mock_page):
    # Setup: JS evaluation returns a voice message object
    mock_page.evaluate.return_value = {'type': 'voice', 'src': 'blob:http://...', 'duration': '0:05', 'text': None}
    result = read_last_message(mock_page)
    assert result['type'] == 'voice'
    assert result['src'] == 'blob:http://...'

def test_capture_full_image_success(mock_page):
    # Setup mocks
    mock_thumb = MagicMock()
    mock_page.evaluate_handle.return_value = mock_thumb
    mock_large_img = MagicMock()
    mock_page.query_selector.return_value = mock_large_img

    filename = "test_image.jpg"

    # Run
    with patch("Autonomous_Reasoning_System.io.wa_multimodal.SAVE_DIR") as mock_dir:
        mock_dir.__truediv__.return_value = "data/incoming_media/test_image.jpg"
        path = capture_full_image(mock_page, filename)

        # Assertions
        assert path == "data/incoming_media/test_image.jpg"
        mock_thumb.click.assert_called_once()
        mock_page.wait_for_selector.assert_called_with('img[src^="blob:"]', timeout=5000)
        mock_large_img.screenshot.assert_called_once()
        mock_page.keyboard.press.assert_called_with("Escape")
        mock_thumb.dispose.assert_called_once()

def test_capture_full_image_no_thumbnail(mock_page):
    mock_page.evaluate_handle.return_value = None
    path = capture_full_image(mock_page, "test.jpg")
    assert path is None

def test_save_voice_note_success(mock_page):
    blob_url = "blob:http://test"
    fake_audio_data = b"fake_audio_data"
    b64_data = base64.b64encode(fake_audio_data).decode('utf-8')
    mock_page.evaluate.return_value = b64_data

    with patch("builtins.open", mock_open()) as mock_file:
        with patch("Autonomous_Reasoning_System.io.wa_multimodal.SAVE_DIR") as mock_dir:
            mock_dir.__truediv__.return_value = "data/incoming_media/wa_123.ogg"

            path = save_voice_note(mock_page, blob_url)

            # Assertions
            assert path == "data/incoming_media/wa_123.ogg"
            mock_file.assert_called_once_with("data/incoming_media/wa_123.ogg", "wb")
            mock_file().write.assert_called_once_with(fake_audio_data)

def test_save_voice_note_fail(mock_page):
    mock_page.evaluate.return_value = None
    path = save_voice_note(mock_page, "blob:fail")
    assert path is None

def test_capture_voice_note_success(mock_page):
    mock_context = MagicMock()
    mock_response = MagicMock()
    mock_response.request.resource_type = "audio"
    mock_response.headers.get.return_value = "audio/ogg"
    mock_response.body.return_value = b"audio_bytes"

    # Mock JS to return true (play button found and clicked)
    mock_page.evaluate.return_value = True

    with patch("builtins.open", mock_open()) as mock_file:
        with patch("Autonomous_Reasoning_System.io.wa_multimodal.SAVE_DIR") as mock_dir:
            mock_dir.__truediv__.return_value = "data/incoming_media/wa_captured.ogg"

            # We need to manually trigger the listener callback
            def side_effect(event, callback):
                if event == "response":
                    callback(mock_response)

            mock_context.on.side_effect = side_effect

            path = capture_voice_note(mock_page, mock_context)

            assert path == "data/incoming_media/wa_captured.ogg"
            mock_file().write.assert_called_once_with(b"audio_bytes")
