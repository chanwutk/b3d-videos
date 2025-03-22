import multiprocessing as mp
import os


IN_DIR = './videos-masked-unsplitted'
OUT_DIR = './videos-masked-splitted'


def split_video(filename: str):
    os.system(f'cat "{os.path.join(OUT_DIR, filename + ".part.*")}" > "{os.path.join(IN_DIR, filename)}"')


def main():
    if os.path.exists(IN_DIR):
        os.system(f'rm -rf "{IN_DIR}"')
    os.makedirs(IN_DIR)

    filenames = set('.'.join(f.split('.')[:-2]) for f in os.listdir(IN_DIR))

    processes: "list[mp.Process]" = []
    for filename in filenames:
        if filename.endswith('.mp4'):
            p = mp.Process(target=split_video, args=(filename,))
            p.start()
            processes.append(p)
    
    for p in processes:
        p.join()
        p.terminate()


if __name__ == '__main__':
    main()