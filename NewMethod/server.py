import asyncio
import json
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
import cv2

pcs = set()

# 각 연결별 PeerConnection과 트랙을 관리하기 위한 dict
connections = {}

async def index(request):
    return web.Response(text="WebRTC 2대 수신 서버 실행 중")

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    pc = RTCPeerConnection()
    pcs.add(pc)
    # 고유 ID 할당 (간단히 id(pc))
    peer_id = id(pc)
    connections[peer_id] = {
        "pc": pc,
        "video_tracks": []
    }
    print(f"새로운 피어 연결: {peer_id}")

    @pc.on("track")
    def on_track(track):
        print(f"[{peer_id}] 트랙 수신: {track.kind}")
        if track.kind == "video":
            connections[peer_id]["video_tracks"].append(track)
            asyncio.create_task(receive_video(track, peer_id))

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)

                if data["type"] == "offer":
                    offer = RTCSessionDescription(
                        sdp=data["sdp"]["sdp"],
                        type=data["sdp"]["type"]
                    )
                    await pc.setRemoteDescription(offer)
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    await ws.send_json({
                        "type": "answer",
                        "sdp": {
                            "type": pc.localDescription.type,
                            "sdp": pc.localDescription.sdp
                        }
                    })
    except Exception as e:
        print(f"[{peer_id}] 오류 발생: {e}")

    await pc.close()
    pcs.discard(pc)
    connections.pop(peer_id, None)
    print(f"[{peer_id}] 피어 연결 종료")
    return ws


async def receive_video(track, peer_id):
    window_name = f"Video from peer {peer_id}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        frame = await track.recv()
        img = frame.to_ndarray(format="bgr24")

        # 창 위치를 다르게 하거나 크기 조정해서 2개 영상 같이 보기 편하게 할 수도 있음
        cv2.imshow(window_name, img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow(window_name)


app = web.Application()
app.router.add_get('/', index)
app.router.add_get('/ws', websocket_handler)

if __name__ == '__main__':
    print("WebRTC 서버 시작: http://localhost:8765")
    web.run_app(app, port=8765)
