import asyncio
import argparse
from  model_manager.model_manager import ModelManager

parser = argparse.ArgumentParser(
    description='Jetson Inference AI Model Manager Server')

#Params for debugging
parser.add_argument("--debug", type=bool, default=False,
                    help='Specify if the server will generate a preformance report of every single executed action')

args = parser.parse_args()

manager = ModelManager()

async def main():
    await manager.start_server(args)

if __name__ == '__main__':
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt: Server shutdown.")