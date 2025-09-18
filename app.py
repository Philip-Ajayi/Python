import os
import asyncio
import json
import base64
import time
import traceback
import io
import logging
from typing import Optional

import websockets
import redis.asyncio as redis
import httpx
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn

from google import genai
from google.genai import types
from PIL import Image
import mss

# ------------------ Logging Setup ------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
)

# ------------------ Redis & HTTP ------------------
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
http_client = httpx.AsyncClient(timeout=30.0)

# ------------------ Config ------------------
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MAX_SESSIONS_PER_MODEL = 3
RECONNECT_TIMEOUT = 300  # seconds to allow frontend to reconnect
QUEUE_TIMEOUT = 60  # max wait time for queue
MONITOR_INTERVAL = 5  # seconds for dashboard updates

# ------------------ Global State ------------------
active_sessions = {}
active_sessions_lock = asyncio.Lock()
queue = asyncio.Queue()
frontend_reconnections = {}  # track reconnections: {session_key: count}
dashboard_connections = []  # active WebSocket connections for dashboard

# ------------------ Utility Functions ------------------
def session_key_for(userid: Optional[str], corporateid: Optional[str]):
    if userid and corporateid:
        return f"user:{corporateid}"
    if userid:
        return f"user:{userid}"
    return None

async def get_user_plan(userid, corporateid=None):
    if corporateid:
        paidplan = int(await r.get(f"user:{corporateid}:paidplan") or 0)
        return paidplan
    paidplan = int(await r.get(f"user:{userid}:paidplan") or 0)
    freetier = int(await r.get(f"user:{userid}:freetier") or 0)
    return paidplan, freetier

async def get_remaining_time(userid, corporateid):
    if corporateid:
        paid = int(await r.get(f"user:{corporateid}:paidplan") or 0)
        return paid
    paid = int(await r.get(f"user:{userid}:paidplan") or 0)
    free = int(await r.get(f"user:{userid}:freetier") or 0)
    return paid + free

# Define Lua script for atomic deduction
ATOMIC_DEDUCT_SCRIPT = """
-- KEYS[1] = free_key (can be nil for corporateid)
-- KEYS[2] = paid_key
-- ARGV[1] = seconds to deduct

local free = 0
if KEYS[1] ~= "" then
    free = tonumber(redis.call("GET", KEYS[1]) or "0")
end

local paid = tonumber(redis.call("GET", KEYS[2]) or "0")
if paid > 0 then
    -- deduct paid first
    local deduct = math.min(paid, tonumber(ARGV[1]))
    redis.call("DECRBY", KEYS[2], deduct)
    return {"paid", deduct}
else
    if KEYS[1] ~= "" then
        local free = tonumber(redis.call("GET", KEYS[1]) or "0")
        if free > 0 then
            local deduct = math.min(free, tonumber(ARGV[1]))
            redis.call("DECRBY", KEYS[1], deduct)
            return {"free", deduct}
        end
    end
    return {"none", 0}
end
"""


# Register it once (globally)
async def register_scripts():
    global ATOMIC_DEDUCT
    ATOMIC_DEDUCT = await r.script_load(ATOMIC_DEDUCT_SCRIPT)

# Modified deduct_time_from_store to ensure Lua script is loaded
async def deduct_time_from_store(userid, corporateid, seconds=1):
    # Ensure Lua script is registered
    global ATOMIC_DEDUCT
    if not 'ATOMIC_DEDUCT' in globals():
        await register_scripts()

    if corporateid:
        free_key = ""  # no free tier for corporate
        paid_key = f"user:{corporateid}:paidplan"
        result = await r.evalsha(ATOMIC_DEDUCT, keys=[free_key, paid_key], args=[seconds])
        return result  # {"paid", deducted_seconds}
    else:
        free_key = f"user:{userid}:freetier"
        paid_key = f"user:{userid}:paidplan"
        result = await r.evalsha(ATOMIC_DEDUCT, keys=[free_key, paid_key], args=[seconds])
        return result

# ------------------ Gemini Key Management ------------------
async def get_gemini_keys():
    resp = await http_client.get("https://www.mivwordhouse.com", follow_redirects=True)

if resp.status_code != 200:
    logging.error(f"Failed to fetch keys: {resp.status_code} {resp.text}")
    return []

try:
    keys = resp.json()
except Exception as e:
    logging.error(f"Invalid JSON response from key service: {e}")
    return []
    if not hasattr(get_gemini_keys, "key_usage"):
        get_gemini_keys.key_usage = {}
        for k in keys:
            get_gemini_keys.key_usage[k] = {
                "models/gemini-2.5-flash-live-001": 0,
                "models/gemini-2.0-flash-live-001": 0,
                "total": 0,
            }
    for k in keys:
        if k not in get_gemini_keys.key_usage:
            get_gemini_keys.key_usage[k] = {
                "models/gemini-2.5-flash-live-001": 0,
                "models/gemini-2.0-flash-live-001": 0,
                "total": 0,
            }
    return keys

async def select_gemini_key_model():
    if not hasattr(get_gemini_keys, "key_usage"):
        await get_gemini_keys()

    usage_map = get_gemini_keys.key_usage

    # prefer least used key that has 2.5 capacity and total<6
    candidates_2_5 = sorted(
        [(k,u) for k,u in usage_map.items() if u["models/gemini-2.5-flash-live-001"] < MAX_SESSIONS_PER_MODEL and u["total"] < (MAX_SESSIONS_PER_MODEL*2)],
        key=lambda kv: kv[1]["models/gemini-2.5-flash-live-001"]
    )
    if candidates_2_5:
        key, usage = candidates_2_5[0]
        usage["models/gemini-2.5-flash-live-001"] += 1
        usage["total"] += 1
        return key, "models/gemini-2.5-flash-live-001"

    # fallback to 2.0 (ensure total < 6 too)
    candidates_2_0 = sorted(
        [(k,u) for k,u in usage_map.items() if u["models/gemini-2.0-flash-live-001"] < MAX_SESSIONS_PER_MODEL and u["total"] < (MAX_SESSIONS_PER_MODEL*2)],
        key=lambda kv: kv[1]["models/gemini-2.0-flash-live-001"]
    )
    if candidates_2_0:
        key, usage = candidates_2_0[0]
        usage["models/gemini-2.0-flash-live-001"] += 1
        usage["total"] += 1
        return key, "models/gemini-2.0-flash-live-001"

    return None, None

async def release_gemini_session(key, model):
    if not key or not model:
        return
    if not hasattr(get_gemini_keys, "key_usage"):
        return
    usage = get_gemini_keys.key_usage.get(key)
    if not usage:
        return
    usage[model] = max(0, usage.get(model, 0) - 1)
    usage["total"] = max(0, usage.get("total", 1) - 1)

# ------------------ Gemini Config ------------------
def create_gemini_config(voice, language, summaries, worktype, industry, languagelearning):
    system_instruction = []
    if summaries:
        safe_summaries = [s for s in summaries if s]
        if safe_summaries:
            system_instruction.append("These are the summary of past session: " + " ; ".join(safe_summaries))
        else:
            system_instruction.append("The user might be a beginner")
    else:
        system_instruction.append("The user might be a beginner")
    if worktype and industry:
        system_instruction.append(f"User works as {worktype} in {industry}")
    if languagelearning:
        system_instruction.append(f"User wants to learn {languagelearning}")

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO",],
        speech_config=types.SpeechConfig(
            language_code=language,
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        ),
        system_instruction=types.Content(parts=[types.Part.from_text(text="; ".join(system_instruction))],
        role="user"),
    )
    return config

# ------------------ Dashboard ------------------
app = FastAPI()

@app.get("/")
async def dashboard():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head><title>Server Dashboard</title></head>
    <body>
        <h1>Gemini Server Dashboard</h1>
        <div>Queue Length: <span id="queue">0</span></div>
        <div>Frontend Reconnections: <span id="reconnects">0</span></div>
        <div>Active Sessions:</div>
        <pre id="sessions">{}</pre>
        <script>
            const ws = new WebSocket("ws://localhost:8766/ws");
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                document.getElementById("queue").textContent = data.queue_length;
                document.getElementById("reconnects").textContent = JSON.stringify(data.reconnections);
                document.getElementById("sessions").textContent = JSON.stringify(data.active_sessions, null, 2);
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html_content)

@app.websocket("/ws")
async def dashboard_ws(websocket: WebSocket):
    origin = websocket.headers.get("origin")
    if origin not in ALLOWED_ORIGINS:
        await websocket.close(code=4003)
        return

    await websocket.accept()
    dashboard_connections.append(websocket)
    try:
        while True:
            await asyncio.sleep(MONITOR_INTERVAL)
    except Exception:
        dashboard_connections.remove(websocket)

async def monitor_dashboard():
    while True:
        data = {
            "active_sessions": {k: {"info": v.get("info"), "started_at": v.get("started_at"), "connected": bool(v.get("frontend_ws"))} for k, v in active_sessions.items()},
            "queue_length": queue.qsize(),
            "reconnections": frontend_reconnections
        }
        for ws in dashboard_connections.copy():
            try:
                await ws.send_text(json.dumps(data))
            except Exception:
                dashboard_connections.remove(ws)
        await asyncio.sleep(MONITOR_INTERVAL)

# ------------------ Queue Handler ------------------
async def queue_handler(websocket, data):
    await queue.put((websocket, data))
    logging.info(f"User queued. Queue length: {queue.qsize()}")
    start_wait = time.time()
    while True:
        key, model = await select_gemini_key_model()
        if key:
            logging.info(f"User assigned key {key} model {model}")
            return key, model
        if time.time() - start_wait > QUEUE_TIMEOUT:
            logging.error("No Gemini API key available: all keys exhausted or max sessions reached.")
            await websocket.send(json.dumps({"error": "all_gemini_keys_exhausted"}))
            await websocket.close()
            return None, None
        await asyncio.sleep(0.5)

# ------------------ Gemini Session ------------------
async def run_gemini_session(key, model, config, initial_frontend_ws, userid, corporateid, session_key):
    client = genai.Client(api_key=key)
    seconds_spent = 0
    session_ctx = active_sessions.get(session_key, {})
    session_ctx.setdefault('started_at', time.time())
    session_ctx.setdefault('info', {'userid': userid, 'corporateid': corporateid})
    session_ctx.setdefault('frontend_ws', initial_frontend_ws)
    session_ctx.setdefault('lock', asyncio.Lock())
    session_ctx.setdefault('waiting_for_reconnect', False)
    session_ctx.setdefault('disconnect_time', None)
    session_ctx.setdefault('deduction_task', None)
    # events to coordinate deduction and forced termination
    gemini_started_event = asyncio.Event()   # set when gemini sends first data
    stop_event = asyncio.Event()             # set when session should end (exhaustion / explicit)
    session_ctx['gemini_started_event'] = gemini_started_event
    session_ctx['stop_event'] = stop_event
    active_sessions[session_key] = session_ctx

    try:
        async with client.aio.live.connect(model=model, config=config) as session:
            start_time = time.time()
            video_out_queue = asyncio.Queue(maxsize=10)

            # deduction controller: runs only when frontend is connected, gemini active, and after first gemini output
            async def deduction_loop():
                # Wait until the first gemini output has occurred
                await gemini_started_event.wait()
                logging.info(f"Deduction loop starting for {session_key} after first gemini output")
                while True:
                    if stop_event.is_set():
                        logging.info(f"Deduction loop noticed stop_event for {session_key}")
                        return
                    if session_key not in active_sessions:
                        logging.info(f"Deduction loop sees session removed: {session_key}")
                        return
                    ctx = active_sessions[session_key]
                    if ctx.get('frontend_ws') is None:
                        await asyncio.sleep(0.5)
                        continue
                    # deduct 1 second
                    try:
                        await deduct_time_from_store(userid, corporateid, 1)
                        remaining = await get_remaining_time(userid, corporateid)
                        if remaining < 0:
                            logging.warning(f"Remaining time negative for session {session_key}: {remaining}")
                            remaining = 0
                        logging.debug(f"After deduct, remaining for {session_key}: {remaining}")
                        if remaining <= 0:
                            logging.info(f"Quota exhausted for {session_key}, flagging stop_event")
                            stop_event.set()
                            return
                    except Exception as e:
                        logging.exception(f"Deduction failure for session {session_key}: {e}")
                    await asyncio.sleep(1)
            # ------------------ In-Memory Quota Deduction ------------------
            # Patch: Ensure in-memory deduction is authoritative and flushed atomically
            async def start_in_memory_deduction(session_key, userid, corporateid):
                ctx = active_sessions.get(session_key)
                if not ctx:
                    return

                ctx['deduction_stop_event'] = asyncio.Event()
                remaining = await get_remaining_time(userid, corporateid)
                ctx['remaining_time'] = remaining

                flush_interval = 60
                last_flush = time.time()

                try:
                    while not ctx['deduction_stop_event'].is_set():
                        await asyncio.sleep(1)
                        # atomic per-second deduction
                        result = await deduct_time_from_store(userid, corporateid, 1)
                        ctx['remaining_time'] -= 1  # keep in-memory counter consistent

                        # flush periodically
                        if time.time() - last_flush >= flush_interval:
                            # Only update Redis if in-memory is ahead
                            await flush_remaining_time_to_redis(userid, corporateid, ctx['remaining_time'])
                            last_flush = time.time()

                        if ctx['remaining_time'] <= 0:
                            if ctx.get('stop_event'):
                                ctx['stop_event'].set()
                            break

                    # final flush
                    await flush_remaining_time_to_redis(userid, corporateid, ctx['remaining_time'])
                except asyncio.CancelledError:
                    await flush_remaining_time_to_redis(userid, corporateid, ctx['remaining_time'])
                    raise


            async def flush_remaining_time_to_redis(userid, corporateid, remaining_seconds):
                if corporateid:
                    paid_key = f"user:{corporateid}:paidplan"
                    await r.set(paid_key, max(0, remaining_seconds))
                else:
                    free_key = f"user:{userid}:freetier"
                    paid_key = f"user:{userid}:paidplan"
                    # calculate free and paid portion
                    free_val = int(await r.get(free_key) or 0)
                    free_remain = min(free_val, remaining_seconds)
                    paid_remain = remaining_seconds - free_remain
                    await r.set(free_key, max(0, free_remain))
                    await r.set(paid_key, max(0, paid_remain))

            # send incoming frontend data to gemini live session
            async def send_from_frontend():
                ctx = active_sessions[session_key]
                while True:
                    if stop_event.is_set():
                        logging.info(f"send_from_frontend stopping due to stop_event for {session_key}")
                        return 'stop'
                    ws = ctx.get('frontend_ws')
                    if ws is None:
                        # wait for reconnection or timeout
                        ctx['waiting_for_reconnect'] = True
                        ctx['disconnect_time'] = time.time()
                        # allow upper-level to handle reconnect wait
                        await asyncio.sleep(0.5)
                        # continue waiting unless stop_event set
                        continue
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=1)
                        data = json.loads(message)
                        # explicit end signal
                        if data.get('action') == 'end_session':
                            logging.info(f"Explicit end received from frontend for {session_key}")
                            stop_event.set()
                            return 'explicit_end'
                        if 'audio' in data:
                            try:
                                await session.send(input={
                                    "data": base64.b64decode(data['audio']),
                                    "mime_type": "audio/pcm"
                                })
                            except Exception as e:
                                logging.exception(f"Failed to send audio to gemini for {session_key}: {e}")
                        if 'text' in data:
                            try:
                                await session.send(input=data['text'])
                            except Exception as e:
                                logging.exception(f"Failed to send text to gemini for {session_key}: {e}")
                        if 'image' in data:
                            try:
                                # push the frame to the video queue instead of sending directly
                                video_out_queue.put_nowait({"mime_type": "image/jpeg", "data": data['image']})
                            except asyncio.QueueFull:
                                logging.warning("Video queue full, skipping frontend frame")
                    except asyncio.TimeoutError:
                        # periodic wake to check stop_event
                        continue
                    except Exception as e:
                        logging.info(f"Frontend websocket recv error for {session_key}: {e}")
                        async with active_sessions_lock:
                            ctx['frontend_ws'] = None
                            frontend_reconnections[session_key] = frontend_reconnections.get(session_key, 0) + 1
                            ctx['waiting_for_reconnect'] = True
                            ctx['disconnect_time'] = time.time()

            # receive gemini output and forward to frontend
            async def forward_to_frontend():
                nonlocal seconds_spent
                ctx = active_sessions[session_key]
                first_output_seen = False
                last_tick = None
                active_seconds = 0

                async for turn in session.receive():
                    if stop_event.is_set():
                        logging.info(f"forward_to_frontend stopping due to stop_event for {session_key}")
                        break

                    out_msg = {}
                    if turn.text:
                        out_msg['text'] = turn.text
                    if turn.data:
                        out_msg['audio'] = base64.b64encode(turn.data).decode()
                    if out_msg:
                        ws = ctx.get('frontend_ws')
                        if ws:
                            try:
                                await ws.send(json.dumps(out_msg))
                            except Exception as e:
                                logging.info(f"Failed to send to frontend for {session_key}: {e}")
                                ctx['frontend_ws'] = None
                                frontend_reconnections[session_key] = frontend_reconnections.get(session_key, 0) + 1
                                ctx['waiting_for_reconnect'] = True
                                ctx['disconnect_time'] = time.time()
                                continue
                    # Start counting active time after first Gemini output
                    if not first_output_seen:
                        gemini_started_event.set()
                        last_tick = time.time()
                        first_output_seen = True

                    # Update active_seconds regardless of frontend disconnection
                    now = time.time()
                    if last_tick:
                        active_seconds += now - last_tick
                    last_tick = now

                seconds_spent = int(active_seconds)



            # video sending (non-blocking capture)
            async def send_video_frames(video_out_queue, stop_event):
                while True:
                    if stop_event.is_set():
                        logging.info("send_video_frames stopping due to stop_event")
                        return
                    frame = await video_out_queue.get()  # queue receives frontend frames
                    try:
                        # send frame to Gemini
                        await session.send(input={
                            "data": base64.b64decode(frame['data']),
                            "mime_type": "image/jpeg"
                        })
                    except Exception as e:
                        logging.exception(f"Failed to send video frame to gemini: {e}")


            # replace:
            # ded_task = asyncio.create_task(deduction_loop())
            # session_ctx['deduction_task'] = ded_task
            ded_task = asyncio.create_task(start_in_memory_deduction(session_key, userid, corporateid))
            async with active_sessions_lock:
                session_ctx['deduction_task'] = ded_task
            # main I/O loop
            end_reason = None
            try:

                # Shield the forward_to_frontend task to prevent losing messages
                io_tasks = [
                    asyncio.create_task(send_from_frontend()),
                    asyncio.create_task(asyncio.shield(forward_to_frontend())),
                    asyncio.create_task(send_video_frames(video_out_queue, stop_event))
                ]

                done, pending = await asyncio.wait(io_tasks, return_when=asyncio.FIRST_COMPLETED)

                for t in pending:
                    if not t.cancelled() and not t.done():
                        t.cancel()
                        try:
                            await t
                        except asyncio.CancelledError:
                            pass

                for t in done:
                    try:
                        res = t.result()
                    except asyncio.CancelledError:
                        res = None
                    if res == 'explicit_end':
                        end_reason = 'explicit_end'
                    elif res == 'disconnected':
                        end_reason = 'disconnected'
                    elif res == 'stop':
                        end_reason = 'stop'
            except Exception as e:
                logging.exception(f"I/O error in session {session_key}: {e}")

            # if disconnected, wait up to RECONNECT_TIMEOUT for frontend to reconnect (unless stop_event set)
            if end_reason == 'disconnected' and not stop_event.is_set():
                logging.info(f"User {session_key} disconnected, waiting up to {RECONNECT_TIMEOUT}s for reconnect...")
                wait_start = time.time()
                reconnected = False
                while time.time() - wait_start < RECONNECT_TIMEOUT:
                    if stop_event.is_set():
                        logging.info(f"Stop event set while waiting for reconnect for {session_key}")
                        break
                    ctx = active_sessions.get(session_key)
                    if not ctx:
                        break
                    if ctx.get('frontend_ws') is not None:
                        reconnected = True
                        ctx['waiting_for_reconnect'] = False
                        ctx['disconnect_time'] = None
                        logging.info(f"User {session_key} reconnected")
                        # resume IO by re-invoking the IO coroutines
                        try:
                            io_tasks = [
                                asyncio.create_task(send_from_frontend()),
                                asyncio.create_task(forward_to_frontend()),
                                asyncio.create_task(send_video_frames(video_out_queue, stop_event))
                            ]
                            done, pending = await asyncio.wait(io_tasks, return_when=asyncio.FIRST_COMPLETED)
                            for t in pending:
                                t.cancel()
                            for t in done:
                                try:
                                    res = t.result()
                                except asyncio.CancelledError:
                                    res = None
                                if res == 'explicit_end':
                                    end_reason = 'explicit_end'
                                    break
                                elif res == 'disconnected':
                                    end_reason = 'disconnected'
                                    break
                                elif res == 'stop':
                                    end_reason = 'stop'
                                    break
                        except Exception:
                            pass
                        if end_reason in ('explicit_end', 'disconnected', 'stop'):
                            break
                    await asyncio.sleep(0.5)
                if not reconnected:
                    logging.info(f"User {session_key} failed to reconnect within timeout")

            # If stop_event is set (quota exhaustion or explicit end), ensure it's noted
            if stop_event.is_set() and not end_reason:
                end_reason = 'stop'

            # Ask Gemini for a final summary (if possible)
            summary_text = ""
            try:
                await session.send(input="Give me summary of what has been discussed", end_of_turn=True)

                async def collect_summary():
                    parts = []
                    async for resp in session.receive():
                        if resp.text:
                            parts.append(resp.text)
                        if getattr(resp, "event", None) == "turn_complete":
                            break
                    return " ".join(parts)

                summary_text = await asyncio.wait_for(collect_summary(), timeout=30.0)
            except asyncio.TimeoutError:
                logging.warning(f"Timeout while waiting for summary from Gemini for {session_key}")
                summary_text = "Summary request timed out."
            except Exception as e:
                logging.exception(f"Failed to get summary from gemini for {session_key}: {e}")
                summary_text = ""


            # cancel deduction task and ensure no more deductions
            # inside finally block in run_gemini_session
            if session_ctx.get('deduction_task'):
                try:
                    session_ctx['deduction_stop_event'].set()
                    session_ctx['deduction_task'].cancel()
                except Exception:
                    pass

            # seconds_spent already being tracked by forward_to_frontend updates; ensure integer
            seconds_spent = int(seconds_spent)

            logging.info(f"User {session_key} session ended. Seconds spent: {seconds_spent}, reason: {end_reason}")

            return summary_text, seconds_spent

    except Exception as e:
        traceback.print_exc()
        return "", seconds_spent
    finally:
        await release_gemini_session(key, model)
        # cleanup active_sessions entry
        async with active_sessions_lock:
            if session_key in active_sessions:
                try:
                    t = active_sessions[session_key].get('deduction_task')
                    if t:
                        t.cancel()
                except Exception:
                    pass
                del active_sessions[session_key]

# ------------------ Frontend Handler ------------------
ALLOWED_ORIGINS = ["http://localhost", "http://127.0.0.1", "https://live.com"]

async def frontend_handler(websocket):
    origin = websocket.request_headers.get("Origin")
    if origin not in ALLOWED_ORIGINS:
        # Reject connection
        await websocket.close(code=4003)  # 4003 = Forbidden
        return

    try:
        data = json.loads(await websocket.recv())

        # --- Required fields ---
        required_fields = ["voicepreference", "languagepreference", "languagelearning", "userid", "summary"]
        for field in required_fields:
            if field not in data or data[field] is None:
                await websocket.send(json.dumps({"error": f"missing_field:{field}"}))
                await websocket.close()
                return

        userid = data.get("userid")
        corporateid = data.get("corporateid")
        worktype = data.get("worktype")
        industry = data.get("industry")

        # --- Rules for corporateid, worktype, industry ---
        if corporateid:
            if not (worktype or industry):
                await websocket.send(json.dumps({"error": "corporateid_requires_worktype_or_industry"}))
                await websocket.close()
                return
        else:
            if worktype or industry:
                await websocket.send(json.dumps({"error": "worktype_or_industry_requires_corporateid"}))
                await websocket.close()
                return

        # Compute session key used to track reconnections
        s_key = session_key_for(userid, corporateid)
        if not s_key:
            await websocket.close()
            return

        # Authorization / plan checks
        if userid and not corporateid:
            paidplan, freetier = await get_user_plan(userid)
            if paidplan == 0 and freetier == 0:
                await websocket.send(json.dumps({"error": "quota_exhausted"}))
                await websocket.close()
                return
        if userid and corporateid:
            paidplan = await get_user_plan(userid, corporateid)
            if paidplan < 144000:
                await websocket.send(json.dumps({"error": "quota_exhausted"}))
                await websocket.close()
                return


        # If there's an existing active session that is waiting for reconnection, attach
        # --- In frontend_handler, before reconnect handling ---
        async with active_sessions_lock:
            existing = active_sessions.get(s_key)
            if existing and existing.get('waiting_for_reconnect'):
                if 'reconnect_lock' not in existing:
                    existing['reconnect_lock'] = asyncio.Lock()
            else:
                existing = None

        reconnect_lock = existing.get('reconnect_lock')
        async with reconnect_lock:
            # Only one coroutine enters this block at a time
            if existing.get('frontend_ws') is not None:
                await websocket.send(json.dumps({"error": "another_reconnect_in_progress"}))
                await websocket.close()
                return

            # Validate payload
            if userid != existing['info'].get('userid') or corporateid != existing['info'].get('corporateid'):
                await websocket.send(json.dumps({"error": "invalid_reconnect"}))
                await websocket.close()
                return

            logging.info(f"Attaching reconnecting websocket to existing session {s_key}")
            existing['frontend_ws'] = websocket
            existing['waiting_for_reconnect'] = False
            existing['disconnect_time'] = None


        try:
            await websocket.wait_closed()
        finally:
            async with active_sessions_lock:
                if active_sessions.get(s_key):
                    active_sessions[s_key]['frontend_ws'] = None

        # otherwise this is a fresh connection: obtain key and start session
        key, model = await queue_handler(websocket, data)
        if not key:
            return

        config = create_gemini_config(
            voice=data.get("voicepreference", "Puck"),
            language=data.get("languagepreference", "en-US"),
            summaries=data.get("summary", []),
            worktype=data.get("worktype"),
            industry=data.get("industry"),
            languagelearning=data.get("languagelearning"),
        )

        # register a minimal session context so reconnections can be attached
        async with active_sessions_lock:
            active_sessions[s_key] = {'frontend_ws': websocket, 'info': {'userid': userid, 'corporateid': corporateid}, 'started_at': time.time()}

        summary_text, seconds_spent = await run_gemini_session(key, model, config, websocket, userid, corporateid, s_key)

        # Post-processing after session end
        languagelearning = data.get('languagelearning')
        async def post_with_retry(url, json_data, retries=3, delay=2):
            for attempt in range(1, retries + 1):
                try:
                    resp = await http_client.post(url, json=json_data)
                    if resp.status_code < 400:
                        return True
                    logging.warning(f"Post attempt {attempt} to {url} failed with status {resp.status_code}")
                except Exception as e:
                    logging.warning(f"Post attempt {attempt} to {url} failed: {e}")
                await asyncio.sleep(delay * attempt)  # exponential backoff
            logging.error(f"Failed to post to {url} after {retries} attempts")
            return False

        # ---- Replacement patch ----
        try:
            if userid and corporateid:
                await post_with_retry(f"https://summary.com/summarysession/{userid}/", {"summary": summary_text, "time": seconds_spent, "language": languagelearning})
                await post_with_retry(f"https://corporate.com/{corporateid}/", {"time": seconds_spent, "languagelearning": languagelearning})
            elif userid and not corporateid:
                await post_with_retry(f"https://summary.com/summarysession/{userid}/", {"summary": summary_text, "time": seconds_spent, "language": languagelearning})
        except Exception as e:
            logging.exception(f"Unexpected error during post-processing for {s_key}: {e}")


    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        traceback.print_exc()
# ------------------ Main ------------------
async def main():
    await get_gemini_keys()
    asyncio.create_task(monitor_dashboard())
    async with websockets.serve(frontend_handler, "0.0.0.0", 8765):
        logging.info("Server started on ws://0.0.0.0:8765")
        config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

if __name__ == "__main__":
    asyncio.run(main())

