import asyncio
import websockets
from server.websocket_server import handle_client


async def main():
    print("WebSocket server is starting...")
    try:

        #Set the host ip
        host = "192.168.178.75"

        # Start the WebSocket server
        start_server = await websockets.serve(handle_client, host , 5000)
        print(f"Server running on ws://{host}.:5000")
        
        await start_server.wait_closed()
        
    except Exception as e:
        print(f"Error starting the server: {e}")
    finally:
        print("Server shutdown.")

if __name__ == '__main__':
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt: Server shutdown.")