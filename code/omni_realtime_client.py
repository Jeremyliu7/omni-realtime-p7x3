# -- coding: utf-8 --

import asyncio
import websockets
import json
import base64
import time
from typing import Optional, Callable, List, Dict, Any
from enum import Enum

class TurnDetectionMode(Enum):
    SERVER_VAD = "server_vad"
    MANUAL = "manual"

DEFAULT_VOICE = "qwen-omni-vc-mm-voice-20260522161248001-f213"

class OmniRealtimeClient:
    """
    与 Omni Realtime API 交互的演示客户端。

    该类提供了连接 Realtime API、发送文本和音频数据、处理响应以及管理 WebSocket 连接的相关方法。

    属性说明:
        base_url (str):
            Realtime API 的基础地址。
        api_key (str):
            用于身份验证的 API Key。
        model (str):
            用于聊天的 Omni 模型名称。
        voice (str):
            服务器合成语音所使用的声音。
        turn_detection_mode (TurnDetectionMode):
            轮次检测模式。
        on_text_delta (Callable[[str], None]):
            文本增量回调函数。
        on_audio_delta (Callable[[bytes], None]):
            音频增量回调函数。
        on_input_transcript (Callable[[str], None]):
            输入转录文本回调函数。
        on_interrupt (Callable[[], None]):
            用户打断回调函数，应在此停止音频播放。
        on_output_transcript (Callable[[str], None]):
            输出转录文本回调函数。
        extra_event_handlers (Dict[str, Callable[[Dict[str, Any]], None]]):
            其他事件处理器，事件名到处理函数的映射。
    """
    def __init__(
        self,
        base_url,
        api_key: str,
        model: str = "",
        voice: str = DEFAULT_VOICE,
        turn_detection_mode: TurnDetectionMode = TurnDetectionMode.MANUAL,
        on_text_delta: Optional[Callable[[str], None]] = None,
        on_audio_delta: Optional[Callable[[bytes], None]] = None,
        on_interrupt: Optional[Callable[[], None]] = None,
        on_input_transcript: Optional[Callable[[str], None]] = None,
        on_output_transcript: Optional[Callable[[str], None]] = None,
        on_conversation_update: Optional[Callable[[str, str], None]] = None,
        extra_event_handlers: Optional[Dict[str, Callable[[Dict[str, Any]], None]]] = None
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.ws = None
        self.on_text_delta = on_text_delta
        self.on_audio_delta = on_audio_delta
        self.on_interrupt = on_interrupt
        self.on_input_transcript = on_input_transcript
        self.on_output_transcript = on_output_transcript
        self.on_conversation_update = on_conversation_update
        self.turn_detection_mode = turn_detection_mode
        self.extra_event_handlers = extra_event_handlers or {}

        # 当前回复状态
        self._current_response_id = None
        self._current_item_id = None
        self._is_responding = False
        # 输入/输出转录打印状态
        self._print_input_transcript = False
        self._output_transcript_buffer = ""
        # 对话文本缓存
        self._current_output_text = ""

    async def connect(self) -> None:
        """与 Realtime API 建立 WebSocket 连接。"""
        url = f"{self.base_url}?model={self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        self.ws = await websockets.connect(
            url, 
            additional_headers=headers,
            ping_interval=20,  # 每20秒发送一次ping
            ping_timeout=10,   # ping超时时间10秒
            close_timeout=10   # 关闭超时时间10秒
        )

        # 设置默认会话配置
        if self.turn_detection_mode == TurnDetectionMode.MANUAL:
            await self.update_session({
                "modalities": ["text", "audio"],
                "voice": self.voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "gummy-realtime-v1"
                },
                "turn_detection" : None
            })
        elif self.turn_detection_mode == TurnDetectionMode.SERVER_VAD:
            await self.update_session({
                "modalities": ["text", "audio"],
                "voice": self.voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "gummy-realtime-v1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.8,
                    "prefix_padding_ms": 500,
                    "silence_duration_ms": 1500
                }
            })
        else:
            raise ValueError(f"Invalid turn detection mode: {self.turn_detection_mode}")

    async def send_event(self, event) -> None:
        event['event_id'] = "event_" + str(int(time.time() * 1000))
        print(f" Send event: type={event['type']}, event_id={event['event_id']}")
        await self.ws.send(json.dumps(event))

    async def update_session(self, config: Dict[str, Any]) -> None:
        """更新会话配置。"""
        event = {
            "type": "session.update",
            "session": config
        }
        print("update session: ", event)
        await self.send_event(event)

    async def stream_audio(self, audio_chunk: bytes) -> None:
        """向 API 流式发送原始音频数据。"""
        # 仅支持 16bit 16kHz 单声道 PCM
        audio_b64 = base64.b64encode(audio_chunk).decode()

        append_event = {
            "type": "input_audio_buffer.append",
            "audio": audio_b64
        }
        await self.send_event(append_event)

    async def commit_audio_buffer(self) -> None:
        """提交音频缓冲区以触发处理。"""
        event = {
            "type": "input_audio_buffer.commit"
        }
        await self.send_event(event)

    async def append_image(self, image_chunk: bytes) -> None:
        """向视频缓冲区追加图像数据。
        图像数据可以来自本地文件，也可以来自实时视频流。

        注意:
            - 图像格式必须为 JPG 或 JPEG。推荐分辨率为 480P 或 720P，最高支持 1080P。
            - 单张图片大小不应超过 500KB。
            - 本方法会将图像数据编码为 Base64 后再发送。
            - 建议以每秒 2 帧的频率向服务器发送图像。
            - 在发送图像数据之前，需要先发送音频数据。
        """
        image_b64 = base64.b64encode(image_chunk).decode()

        event = {
            "type": "input_image_buffer.append",
            "image": image_b64
        }
        await self.send_event(event)

    async def create_response(self) -> None:
        """向 API 请求生成回复（仅在手动模式下需要调用）。"""
        event = {
            "type": "response.create",
            "response": {
                "instructions": """系统提示（S0阶段催收专员电话场景版）
你是海尔消费金融的催收专员，负责逾期1-3天的客户。
你的目标是尽量让用户今天一次性结清欠款，态度始终保持温和、礼貌、专业。
请直接称呼用户为您，避免称呼先生或者女士。
请不要核身。
用户表示没钱时，可以建议找亲朋好友周转。
用户表示明天还时，可以说明天可能存在罚息。
不得建议用户申请、等待或暗示存在任何部分还款、宽限、分期、延期、减免可能。
关于征信：施压可以说（“持续逾期可能会影响个人征信”，“一旦征信受损，会影响未来办理信用卡、房贷”。“征信不良记录会直接影响您求职、租房、出国”，“持续逾期不仅产生罚息，更可能导致征信受损，未来购房、购车贷款审批难度增加”，“若持续逾期超过一定时间仍未还款，系统将标记为潜在风险，并可能上报征信。”，“持续逾期将被金融机构视为还款意愿或能力不足，进而影响您在所有平台的使用评级。”，“持续逾期可能导致征信被划到不良逾期的等级，银行可能因此降低您的授信额度”等），不得暗示或明示“今天不还就会上征信”，也不能承诺具体多少天后会上报征信。避免因绝对的说法引发客户投诉。
关于还款渠道：可以使用海尔消费金融官方的「够花」APP还款。
不得建议用户转账到个人，来门店还款、截图发给个人等行为。
用户过激情绪，请先安抚情绪。
只能与还款提醒相关并且不能使用容易激起客户反感和投诉的话语。
客户询问身份可以说这边是海尔消费金融公司的工作人员，负责您的还款提醒。禁止说自己是真人。
对于有还款意愿和沟通意愿的用户不要用征信或罚息施压。
当客户表示在忙或者要求晚点再打电话
礼貌结束，回复：好的，我们稍后再联系您，请保持手机畅通，再见。
结束时，加一个再见。
记住，非常重要，请主动引导用户，适当添加反问。
用户本次逾期信息如下：
用户称呼：张三
性别：男
逾期金额：1000
逾期天数：1
当前时间：20260521
对话风格要求（非常重要）：
每次只说一个意思，一句话结束。
句子要简短、口语化，像真人打电话一样自然。
控制每句话在50字以内，避免长句。
不要一次性列举多个理由或方案，等用户回应后再讲下一个点。""",
                "modalities": ["text", "audio"]
            }
        }
        print("create response: ", event)
        await self.send_event(event)

    async def cancel_response(self) -> None:
        """取消当前回复。"""
        event = {
            "type": "response.cancel"
        }
        await self.send_event(event)

    async def handle_interruption(self):
        """处理用户对当前回复的打断。"""
        if not self._is_responding:
            return

        print(" Handling interruption")

        # 1. 取消当前回复
        if self._current_response_id:
            await self.cancel_response()

        self._is_responding = False
        self._current_response_id = None
        self._current_item_id = None

    async def handle_messages(self) -> None:
        try:
            async for message in self.ws:
                event = json.loads(message)
                event_type = event.get("type")
                
                if event_type != "response.audio.delta":
                    print(" event: ", event)
                else:
                    print(" event_type: ", event_type)

                if event_type == "error":
                    print(" Error: ", event['error'])
                    continue
                elif event_type == "response.created":
                    self._current_response_id = event.get("response", {}).get("id")
                    self._is_responding = True
                elif event_type == "response.output_item.added":
                    self._current_item_id = event.get("item", {}).get("id")
                elif event_type == "response.done":
                    self._is_responding = False
                    self._current_response_id = None
                    self._current_item_id = None
                # Handle interruptions
                elif event_type == "input_audio_buffer.speech_started":
                    print(" Speech detected")
                    if self._is_responding:
                        print(" Handling interruption")
                        await self.handle_interruption()

                    if self.on_interrupt:
                        print(" Handling on_interrupt, stop playback")
                        self.on_interrupt()
                elif event_type == "input_audio_buffer.speech_stopped":
                    print(" Speech ended")
                # Handle normal response events
                elif event_type == "response.text.delta":
                    if self.on_text_delta:
                        self.on_text_delta(event["delta"])
                elif event_type == "response.audio.delta":
                    if self.on_audio_delta:
                        audio_bytes = base64.b64decode(event["delta"])
                        self.on_audio_delta(audio_bytes)
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = event.get("transcript", "")
                    if self.on_input_transcript:
                        await asyncio.to_thread(self.on_input_transcript,transcript)
                        self._print_input_transcript = True
                    # 发送用户文本到对话更新回调
                    if self.on_conversation_update:
                        await asyncio.to_thread(self.on_conversation_update, "user", transcript)
                elif event_type == "response.audio_transcript.done":
                    transcript = event.get("transcript", "")
                    if self.on_output_transcript:
                        await asyncio.to_thread(self.on_output_transcript,transcript)
                        # self._print
                    if self.on_conversation_update:
                        await asyncio.to_thread(self.on_conversation_update, "assistant", transcript)

                elif event_type == "response.audio_transcript.delta":
                    pass
                    # if self.on_output_transcript:
                    #     delta = event.get("delta", "")
                    #     if not self._print_input_transcript:
                    #         self._output_transcript_buffer += delta
                    #     else:
                    #         if self._output_transcript_buffer:
                    #             await asyncio.to_thread(self.on_output_transcript,self._output_transcript_buffer)
                    #             self._output_transcript_buffer = ""
                    #         await asyncio.to_thread(self.on_output_transcript,delta)
                    # # 累积完整的输出文本
                    # delta = event.get("delta", "")
                    # self._current_output_text += delta
                elif event_type == "response.audio_transcript.done":
                    self._print_input_transcript = False
                    # 发送完整的AI回复文本到对话更新回调
                    if self.on_conversation_update and self._current_output_text.strip():
                        await asyncio.to_thread(self.on_conversation_update, "assistant", self._current_output_text.strip())
                    # 清空当前输出文本缓存
                    self._current_output_text = ""
                elif event_type in self.extra_event_handlers:
                    self.extra_event_handlers[event_type](event)

        except websockets.exceptions.ConnectionClosed:
            print(" Connection closed")
        except Exception as e:
            print(" Error in message handling: ", str(e))

    async def close(self) -> None:
        """关闭 WebSocket 连接。"""
        if self.ws:
            await self.ws.close()