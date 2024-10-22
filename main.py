import asyncio
import websockets
from server.websocket_server import handle_client

# Start the WebSocket server
start_server = websockets.serve(handle_client, "192.168.178.75", 5000)

if __name__ == '__main__':
    print("WebSocket server is starting...")
    try:
        asyncio.get_event_loop().run_until_complete(start_server)
        print("Server running on ws://192.168.178.75:5000")
        asyncio.get_event_loop().run_forever()
    except Exception as e:
        print(f"Error starting the server: {e}")
    finally:
        print("Server shutdown.")