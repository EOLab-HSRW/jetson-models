import shutil
import asyncio
import argparse
from pathlib import Path
from  model_manager.model_manager import ModelManager

parser = argparse.ArgumentParser(
    description='Jetson Inference AI Model Manager Server')

#Params for debugging
parser.add_argument("--debug", type=bool, default=False,
                    help='Specify if the server will generate a preformance report of every single executed action')

#Params for dataset
parser.add_argument("--delete-datasets", type=bool, default=False,
                    help='Specify if the server will delete all the created datasets after the server is closed')

args = parser.parse_args()

manager = ModelManager()

async def main():
    await manager.start_server(args)

if __name__ == '__main__':
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        dataset_path = Path("datasets")
        if args.delete_datasets and Path(dataset_path).is_dir():
            shutil.rmtree(dataset_path)
            print('\nDataset folder and its content removed')
        print("\nKeyboardInterrupt: Server shutdown.")