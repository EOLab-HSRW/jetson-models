import asyncio
from  model_manager.model_manager import ModelManager

manager = ModelManager()

async def main():
    await manager.start_server()

if __name__ == '__main__':
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt: Server shutdown.")