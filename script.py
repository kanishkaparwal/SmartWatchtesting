#!/usr/bin/env python3
"""
script.py

Flow:
  - adb screenshot -> /sdcard/screen.png -> pull to host
  - send screenshot to Vision API to locate requested icon text (e.g. "flashlight")
  - parse response for label bounding box -> compute center x,y
  - adb input tap x y
  - move to next command

Notes:
  - If Vision API fails to find the icon, there's a fallback with OpenCV template matching (if you provide a template image).
  - Accessibility enabling is attempted only if you provide a service id. Enabling requires the service to already be installed as an AccessibilityService.

Adapt the `send_to_vision` parser to match the actual API response format.
"""

import os
import subprocess
import time
import base64
import json
from typing import Optional, Tuple, Dict, List

import requests
from PIL import Image
import numpy as np
import cv2

# ---- CONFIG ----
ADB = "adb"  # or full path to adb
REMOTE_SCREENSHOT = "/sdcard/automation_screen.png"
LOCAL_SCREENSHOT = "screen.png"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "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")
OPENAI_VISION_ENDPOINT = "https://api.openai.com/v1/chat/completions"  # placeholder; adapt to actual endpoint
VISION_MODEL = "gpt-4o"  # placeholder model name; change if needed

# Confidence and retry settings
VISION_CONFIDENCE_THRESHOLD = 0.6
MAX_RETRIES = 2
WAIT_AFTER_TAP = 0.8  # seconds

# Optional: mapping of user commands -> search labels the vision model understands
LABEL_ALIASES = {
    "tap on flashlight icon": ["flashlight", "torch", "flash light"],
    "tap on calendar icon": ["calendar", "date", "agenda"],
    # Add more mappings as you discover label vocabulary
}

# Accessibility service id (set to None to skip enabling)
ACCESSIBILITY_SERVICE_ID = None  # e.g. "com.myorg.myapp/.MyAccessibilityService"

# ---- Utility: run adb command ----
def run_adb(args: List[str], timeout: int = 10) -> Tuple[int, str, str]:
    cmd = [ADB] + args
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate(timeout=timeout)
    return proc.returncode, out.strip(), err.strip()

# ---- Screenshot functions ----
def take_device_screenshot(local_path: str = LOCAL_SCREENSHOT) -> bool:
    """Takes screenshot on device and pulls to host."""
    rc, out, err = run_adb(["shell", "screencap", "-p", REMOTE_SCREENSHOT])
    if rc != 0:
        print("adb screencap failed:", err or out)
        return False
    rc, out, err = run_adb(["pull", REMOTE_SCREENSHOT, local_path])
    if rc != 0:
        print("adb pull failed:", err or out)
        return False
    # optional: remove remote screenshot
    run_adb(["shell", "rm", REMOTE_SCREENSHOT])
    return True

# ---- Accessibility handling (best-effort) ----
def is_accessibility_enabled() -> bool:
    rc, out, err = run_adb(["shell", "settings", "get", "secure", "accessibility_enabled"])
    if rc != 0:
        print("Error checking accessibility_enabled:", err or out)
        return False
    return out.strip() == "1"

def get_enabled_accessibility_services() -> str:
    rc, out, err = run_adb(["shell", "settings", "get", "secure", "enabled_accessibility_services"])
    if rc != 0:
        return ""
    return out.strip()

def enable_accessibility_service(service_id: str) -> bool:
    """
    Best-effort: set enabled_accessibility_services to include service_id and set accessibility_enabled=1.
    WARNING: This modifies device settings and may require appropriate device privileges.
    """
    if not service_id:
        return False

    current = get_enabled_accessibility_services()
    if service_id in current:
        print("Service already in enabled_accessibility_services.")
    else:
        new_val = service_id if not current or current.lower() == "null" else current + ":" + service_id
        # write the new list
        rc, out, err = run_adb(["shell", "settings", "put", "secure", "enabled_accessibility_services", new_val])
        if rc != 0:
            print("Failed to put enabled_accessibility_services:", err or out)
            return False

    # enable accessibility globally
    rc, out, err = run_adb(["shell", "settings", "put", "secure", "accessibility_enabled", "1"])
    if rc != 0:
        print("Failed to enable accessibility:", err or out)
        return False
    # verify
    return is_accessibility_enabled()

# ---- Vision API call (fixed for OpenAI Vision models) ----
def send_to_vision(image_path: str, prompt: str) -> dict:
    """
    Uses OpenAI's GPT-4o (vision-capable) model to find element coordinates in a screenshot.
    """
    with open(image_path, "rb") as f:
        img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
    "model": "gpt-4o",  # or "gpt-4o-mini"
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""
You are identifying smartwatch app icons from a circular Wear OS home screen screenshot.
Locate the icon described below, and return its bounding box coordinates (x, y, width, height)
for the center of the icon.

Description: "{prompt}"

Return ONLY a valid JSON object in the format:
{{"bbox":[x,y,width,height],"confidence":float}}.

If you cannot find it, respond with:
{{"bbox":null,"confidence":0.0}}.
"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}"
                    }
                }
            ]
        }
    ],
    "max_tokens": 300
}

    resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    resp.raise_for_status()
    result = resp.json()

    # Extract model text output (should be JSON)
    content = result["choices"][0]["message"]["content"].strip()
    print("🧠 Vision API raw output:", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0).replace("'", '"'))
        return {}




# ---- Parse Vision response (adapt this to your response format) ----
def find_label_bbox_from_vision_response(resp_json: dict, label_candidates: List[str]) -> Optional[Tuple[int, int, int, int, float]]:
    """
    Handles both simple JSON {"bbox":[x,y,w,h],"confidence":float}
    and detailed responses with predictions/objects.
    """
    # Case 1: simple direct bbox + confidence
    if "bbox" in resp_json and "confidence" in resp_json:
        x, y, w, h = resp_json["bbox"]
        conf = float(resp_json["confidence"])
        return (int(x), int(y), int(w), int(h), conf)

    # Case 2: more complex prediction list
    preds = resp_json.get("predictions") or resp_json.get("objects") or []
    for p in preds:
        label = str(p.get("label", "")).lower()
        conf = float(p.get("confidence", 1.0))
        bbox = p.get("bbox")
        if bbox and len(bbox) == 4:
            x, y, w, h = bbox
            return (int(x), int(y), int(w), int(h), conf)

    return None


# ---- Fallback: OpenCV template matching ----
def template_match(image_path: str, template_path: str, threshold: float = 0.7) -> Optional[Tuple[int,int,int,int,float]]:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    tpl = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if img is None or tpl is None:
        print("Template or image not found for template matching.")
        return None
    ih, iw = img.shape[:2]
    th, tw = tpl.shape[:2]
    res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
    minv, maxv, minloc, maxloc = cv2.minMaxLoc(res)
    if maxv >= threshold:
        x, y = maxloc
        return (x, y, tw, th, float(maxv))
    return None

# ---- ADB tap ----
def adb_tap(x: int, y: int):
    rc, out, err = run_adb(["shell", "input", "tap", str(int(x)), str(int(y))])
    if rc != 0:
        raise RuntimeError(f"adb tap failed: {err or out}")
    time.sleep(WAIT_AFTER_TAP)

# ---- Main runner for one command ----
def perform_command(command_text: str, template_path: Optional[str] = None) -> bool:
    """
    command_text: e.g. "tap on flashlight icon"
    template_path: optional fallback template image file to template-match
    """
    print(f"\n--- Executing command: '{command_text}' ---")
    # Take screenshot
    ok = take_device_screenshot()
    if not ok:
        print("Failed to take screenshot. Aborting command.")
        return False

    prompt = f"Locate the UI element that matches: {command_text}. Return bounding boxes with labels and confidence."

    # Try Vision API (with retries)
    resp_json = None
    for attempt in range(MAX_RETRIES):
        try:
            resp_json = send_to_vision(LOCAL_SCREENSHOT, prompt)
            break
        except Exception as e:
            print(f"Vision API attempt {attempt+1} failed: {e}")
            time.sleep(1)
    if resp_json:
        # Look up label candidates from LABEL_ALIASES
        candidates = LABEL_ALIASES.get(command_text.lower(), [command_text.replace("tap on ", "").replace(" icon","").strip()])
        bbox = find_label_bbox_from_vision_response(resp_json, candidates)
        if bbox and bbox[4] >= VISION_CONFIDENCE_THRESHOLD:
            x, y, w, h, conf = bbox
            cx = x + w // 2
            cy = y + h // 2
            print(f"Found via Vision: label~ bbox={x,y,w,h} confidence={conf:.2f} -> tapping ({cx},{cy})")
            adb_tap(cx, cy)
            return True
        else:
            print("Vision API did not return a confident bbox.")

    # Fallback: template matching if template provided
#    if template_path:
#        found = template_match(LOCAL_SCREENSHOT, template_path, threshold=0.65)
#        if found:
#            x, y, w, h, score = found
#            cx, cy = x + w//2, y + h//2
#            print(f"Template matched at ({x},{y},{w},{h}) score={score:.2f} -> tapping ({cx},{cy})")
#            adb_tap(cx, cy)
#            return True
#        else:
#            print("Template matching didn't find the element.")

    # Optionally, you can add heuristics (OCR + text match), color/shape detection etc.
    print(f"Unable to find '{command_text}'.")
    return False

# ---- Example flow for list of commands ----
def run_commands(commands: List[Dict]):
    """
    commands: list of dicts: {"cmd": "tap on flashlight icon", "template": "templates/flashlight.png" (optional)}
    """
    # Handle accessibility if requested
    if ACCESSIBILITY_SERVICE_ID:
        print("Checking accessibility state...")
        enabled = is_accessibility_enabled()
        if not enabled:
            print("Accessibility not enabled. Attempting to enable the provided service id...")
            ok = enable_accessibility_service(ACCESSIBILITY_SERVICE_ID)
            print("Enabled accessibility:", ok)
        else:
            print("Accessibility already enabled.")

    for c in commands:
        cmd_text = c["cmd"]
        tpl = c.get("template")
        success = perform_command(cmd_text, tpl)
        if not success:
            print(f"[WARN] command failed: {cmd_text}")
        # small delay between commands
        time.sleep(0.5)

# ---- If run as main ----
if __name__ == "__main__":
    # Example command list
    cmds = [
        {"cmd": "tap on flashlight icon", "template": "templates/flashlight.png"},
    ]
    run_commands(cmds)

