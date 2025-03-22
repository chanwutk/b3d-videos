import multiprocessing as mp
import os


IN_DIR = './videos-masked'
OUT_DIR = './videos-masked-splitted'


def split_video(filename: str):
    os.system(f'split -d -b 50M "{os.path.join(IN_DIR, filename)}" "{os.path.join(OUT_DIR, filename + ".part.")}"')


def main():
    if os.path.exists(OUT_DIR):
        os.system(f'rm -rf "{OUT_DIR}"')
    os.makedirs(OUT_DIR)

    processes: "list[mp.Process]" = []
    for filename in os.listdir(IN_DIR):
        if filename.endswith('.mp4'):
            p = mp.Process(target=split_video, args=(filename,))
            p.start()
            processes.append(p)
    
    for p in processes:
        p.join()
        p.terminate()


if __name__ == '__main__':
    main()