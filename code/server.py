import os
import time
import base64
import asyncio
import logging
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from omni_realtime_client import OmniRealtimeClient, TurnDetectionMode

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 重连配置
MAX_RECONNECT_ATTEMPTS = 5
INITIAL_RECONNECT_DELAY = 1.0  # 秒
MAX_RECONNECT_DELAY = 30.0  # 秒
RECONNECT_BACKOFF_FACTOR = 2.0

# WebSocket心跳配置
HEARTBEAT_INTERVAL = 25  # 每25秒发送一次心跳
WEBSOCKET_TIMEOUT = 60   # WebSocket接收超时时间调整为60秒

class ReconnectManager:
    def __init__(self):
        self.reconnect_attempts = 0
        self.last_reconnect_time = 0
        self.is_reconnecting = False
    
    def should_reconnect(self, error_code: int = None) -> bool:
        """检查是否应该重连"""
        if self.reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
            logger.error(f"达到最大重连次数 {MAX_RECONNECT_ATTEMPTS}，停止重连")
            return False
        
        # 专门检测1011错误代码
        if error_code == 1011:
            logger.warning(f"检测到1011内部服务器错误，准备重连 (尝试 {self.reconnect_attempts + 1}/{MAX_RECONNECT_ATTEMPTS})")
            return True
        
        # 也可以处理其他错误代码
        if error_code in [1006, 1011, 1012, 1013, 1014, 1015]:
            logger.warning(f"检测到错误代码 {error_code}，准备重连 (尝试 {self.reconnect_attempts + 1}/{MAX_RECONNECT_ATTEMPTS})")
            return True
        
        return False
    
    def get_reconnect_delay(self, error_code: int = None) -> float:
        """获取重连延迟时间（指数退避）"""
        if error_code == 1011:
            logger.info("检测到1011错误，立即重连")
            return 0.0
            
        delay = min(INITIAL_RECONNECT_DELAY * (RECONNECT_BACKOFF_FACTOR ** self.reconnect_attempts), MAX_RECONNECT_DELAY)
        logger.info(f"重连延迟: {delay:.2f}秒")
        return delay
    
    def increment_attempts(self):
        """增加重连尝试次数"""
        self.reconnect_attempts += 1
        self.last_reconnect_time = time.time()
    
    def reset(self):
        """重置重连状态"""
        self.reconnect_attempts = 0
        self.last_reconnect_time = 0
        self.is_reconnecting = False
        logger.info("重连状态已重置")

async def create_and_connect_client(
    api_key: str, 
    on_audio_callback, 
    on_interrupt_callback, 
    on_input_transcript_callback,
    on_output_transcript_callback,
    voice: str
) -> OmniRealtimeClient:
    """创建并连接OmniRealtimeClient"""
    client = OmniRealtimeClient(
        base_url="wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
        api_key=api_key,
        model="qwen3-omni-flash-realtime-2025-12-01",
        voice=voice,
        on_audio_delta=lambda d: asyncio.create_task(on_audio_callback(d)),
        on_interrupt=lambda: asyncio.create_task(on_interrupt_callback()),
        on_input_transcript=on_input_transcript_callback,
        on_output_transcript=on_output_transcript_callback,
        turn_detection_mode=TurnDetectionMode.SERVER_VAD,
    )
    
    await client.connect()
    logger.info(f"OmniRealtimeClient连接成功, 使用音色: {voice}")
    return client

async def stream_video_data_task(client: OmniRealtimeClient, video_queue: asyncio.Queue):
    """从队列中获取视频数据并发送"""
    while True:
        try:
            frame = await video_queue.get()
            if frame is None:
                break
            
            # 编码为JPEG格式
            # _, buffer = cv2.imencode('.jpg', frame)
            # image_bytes = buffer.tobytes()

            await client.append_image(frame)
            logger.debug(f"发送视频数据: {len(frame)} bytes")
            video_queue.task_done()
        except Exception as e:
            logger.error(f"发送视频数据时出错: {e}")

async def send_heartbeat(websocket: WebSocket):
    """发送心跳包任务"""
    while True:
        try:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            await websocket.send_text("ping")
            logger.debug("发送心跳包")
        except Exception as e:
            logger.error(f"发送心跳包失败: {e}")
            break

@app.get("/")
async def get():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except Exception as e:
        logger.error(f"读取index.html失败: {e}")
        return HTMLResponse("<h1>服务器错误</h1>", status_code=500)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, voice: str = "Ethan"):
    await websocket.accept()
    logger.info(f"WebSocket连接已建立, 音色: {voice}")
    
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        logger.error("DASHSCOPE_API_KEY环境变量未设置")
        await websocket.close(code=1008, reason="API密钥未配置")
        return
    
    reconnect_manager = ReconnectManager()
    client = None
    message_task = None
    video_sender_task = None
    websocket_active = True
    video_queue = asyncio.Queue()
    # 获取当前事件循环（用于跨线程回调）
    loop = asyncio.get_running_loop()

    # Start heartbeat task once
    heartbeat_task = asyncio.create_task(send_heartbeat(websocket))
    logger.info("心跳任务已启动")
    
    async def on_audio(data: bytes):
        nonlocal websocket_active
        if not websocket_active: return
        try:
            await websocket.send_bytes(data)
            logger.debug(f"发送音频数据: {len(data)} bytes")
        except Exception as e:
            logger.error(f"发送音频数据失败: {e}")
            websocket_active = False
    
    async def on_interrupt():
        nonlocal websocket_active
        if not websocket_active: return
        try:
            await websocket.send_text("interrupt")
            logger.info("发送中断信号")
        except Exception as e:
            logger.error(f"发送中断信号失败: {e}")
            websocket_active = False

    # -------- 录音转录 & 回复文本回调 --------
    def on_input_transcript(transcript: str):
        nonlocal websocket_active
        if not websocket_active:
            return
        try:
            # 将发送操作调度回主事件循环
            asyncio.run_coroutine_threadsafe(
                websocket.send_json({"type": "input_transcript", "data": transcript}),
                loop
            )
            logger.info(f"发送输入转录: {transcript}")
        except Exception as e:
            logger.error(f"发送输入转录失败: {e}")
            websocket_active = False

    def on_output_transcript(transcript: str):
        nonlocal websocket_active
        if not websocket_active:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                websocket.send_json({"type": "output_transcript", "data": transcript}),
                loop
            )
            logger.info(f"发送输出转录: {transcript}")
        except Exception as e:
            logger.error(f"发送输出转录失败: {e}")
            websocket_active = False

    try:
        while websocket_active:
            try:
                # 1. Create client if it doesn't exist
                if client is None:
                    logger.info("尝试创建并连接OmniRealtimeClient...")
                    client = await create_and_connect_client(api_key, on_audio, on_interrupt, on_input_transcript, on_output_transcript, voice)
                    message_task = asyncio.create_task(client.handle_messages())
                    video_sender_task = asyncio.create_task(stream_video_data_task(client, video_queue))
                    reconnect_manager.reset()
                    logger.info("OmniRealtimeClient已连接并准备就绪")

                # 2. Main message processing loop
                while websocket_active:
                    message = await asyncio.wait_for(websocket.receive(), timeout=WEBSOCKET_TIMEOUT)
                    
                    if message['type'] == 'websocket.receive':
                        if 'bytes' in message:
                            data = message['bytes']
                            if not data: continue
                            
                            stream_type = data[0]
                            content = data[1:]
                            
                            if stream_type == 0:  # audio
                                encoded = base64.b64encode(content).decode("utf-8")
                                event = {
                                    "event_id": "e" + str(int(time.time() * 1000)),
                                    "type": "input_audio_buffer.append",
                                    "audio": encoded,
                                }
                                await client.send_event(event)
                                logger.debug(f"发送音频数据到模型: {len(content)} bytes")
                            elif stream_type == 1:  # video
                                await video_queue.put(content)
                                logger.debug(f"视频数据已入队: {len(content)} bytes")

                        elif 'text' in message:
                            text_data = message['text']
                            if text_data == "pong":
                                logger.debug("收到心跳回应")
                            else:
                                logger.info(f"收到文本消息: {text_data}")

                    elif message['type'] == 'websocket.disconnect':
                        logger.info("收到断开连接消息")
                        websocket_active = False
                        break 
                else: # TimeoutError
                    logger.debug("WebSocket接收超时，继续等待...")
                    continue 
            
            except Exception as e:
                logger.error(f"处理数据时发生错误: {e}")
                
                error_code = getattr(e, 'code', None)
                if hasattr(e, 'args') and len(e.args) > 0:
                    error_str = str(e.args[0])
                    if '1011' in error_str:
                        error_code = 1011
                
                if reconnect_manager.should_reconnect(error_code):
                    reconnect_manager.increment_attempts()
                    delay = reconnect_manager.get_reconnect_delay(error_code)
                    
                    # Clean up for reconnection
                    if message_task and not message_task.done():
                        message_task.cancel()
                    if video_sender_task:
                        await video_queue.put(None)
                        video_sender_task.cancel()
                    if client:
                        await client.close()
                        client = None
                    
                    if delay > 0:
                        logger.info(f"等待 {delay:.2f}秒后重连...")
                        await asyncio.sleep(delay)
                    continue 
                else:
                    logger.error("发生不可恢复的错误或达到最大重连次数，关闭连接")
                    websocket_active = False
                    break

    finally:
        logger.info("开始最终清理...")
        if heartbeat_task and not heartbeat_task.done():
            heartbeat_task.cancel()
        if message_task and not message_task.done():
            message_task.cancel()
        if video_sender_task and not video_sender_task.done():
            video_sender_task.cancel()
        if client:
            try:
                await client.close()
            except Exception:
                pass

        logger.info("清理完成，连接已关闭")


def run_server():
    """启动服务器"""
    logger.info("启动HTTP服务器...")
    # 在函数计算环境中，HTTPS由API网关处理，应用本身只需要HTTP
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info")

if __name__ == "__main__":
    run_server() 