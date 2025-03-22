import json
import os
import shutil
from xml.etree import ElementTree

import cv2
from matplotlib.path import Path
import numpy as np
import torch


IN_DIR = './videos'
OUT_DIR = './videos-masked'


def get_bitmap(width: int, height: int, mask: ElementTree.Element):
    domain = mask.find('.//polygon[@label="domain"]')
    assert domain is not None

    domain = domain.attrib['points']
    domain = domain.replace(';', ',')
    domain = np.array([
        float(pt) for pt in domain.split(',')]).reshape((-1, 2))
    tl = (int(np.min(domain[:, 1])), int(np.min(domain[:, 0])))
    br = (int(np.max(domain[:, 1])), int(np.max(domain[:, 0])))
    domain_poly = Path(domain)
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x, y = x.flatten(), y.flatten()
    pixel_points = np.vstack((x, y)).T
    bitmap = domain_poly.contains_points(pixel_points)
    bitmap = bitmap.reshape((height, width, 1))

    bitmap = bitmap[tl[0]:br[0], tl[1]:br[1], :]
    return bitmap, tl, br


def logger(queue: "torch.multiprocessing.Queue"):
    with open('crop.jsonl', 'w') as fc:
        while True:
            val = queue.get()
            if val == None:
                return
            fc.write(val + '\n')
            fc.flush()


def process(gpuIdx: int, file: str, mask: "ElementTree.Element", logger_queue: "torch.multiprocessing.Queue"):
    cap = cv2.VideoCapture(os.path.join(IN_DIR, file))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mask_img = mask.find(f'.//image[@name="{file.split(".")[0]}.jpg"]')
    assert mask_img is not None, (mask_img, file)

    bitmask, mtl, mbr = get_bitmap(width, height, mask_img)
    logger_queue.put(json.dumps({'tl': mtl, 'br': mbr, 'file': file}))
    bitmask = torch.from_numpy(bitmask).to(f'cuda:{gpuIdx}').to(torch.bool)

    width, height = mbr[1] - mtl[1], mbr[0] - mtl[0]
    writer = cv2.VideoWriter(os.path.join(OUT_DIR, file), cv2.VideoWriter.fourcc(*'mp4v'), 30, (width, height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = torch.from_numpy(frame).to(f'cuda:{gpuIdx}')[mtl[0]:mbr[0], mtl[1]:mbr[1], :] * bitmask
        assert (height, width) == frame.shape[:2], (height, width, frame.shape[:2])

        frame = frame.detach().cpu().numpy()
        frame = frame.astype(np.uint8)

        writer.write(frame)

    writer.release()
    cap.release()

    # filename = os.path.join(OUT_DIR, file)
    # filename_x264 = f"{filename[:-len('.mp4')]}.x264.mp4"

    # if os.path.exists(filename_x264):
    #     os.remove(filename_x264)

    # command = (
    #     "docker run --rm -v $(pwd):/config linuxserver/ffmpeg " +
    #     "-i {input_file} ".format(input_file=os.path.join('/config', filename)) +
    #     "-vcodec libx264 " +
    #     "{output_file}".format(output_file=os.path.join('/config', filename_x264))
    # )
    # os.system(command)


def main():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.mkdir(OUT_DIR)

    fc = open('crop.jsonl', 'w')

    tree = ElementTree.parse('masks.xml')
    mask = tree.getroot()

    num_cuda = torch.cuda.device_count()

    logger_queue = torch.multiprocessing.Queue()
    logger_process = torch.multiprocessing.Process(target=logger, args=(logger_queue,))
    logger_process.start()

    ps: list[torch.multiprocessing.Process] = []
    for fidx, file in enumerate(os.listdir(IN_DIR)):
        p = torch.multiprocessing.Process(target=process, args=(fidx % num_cuda, file, mask, logger_queue))
        p.start()
        ps.append(p)

    for p in ps:
        p.join()
        p.terminate()

    logger_queue.put(None)
    logger_process.join()
    logger_process.terminate()
    cv2.destroyAllWindows()
    fc.close()


if __name__ == '__main__':
    main()