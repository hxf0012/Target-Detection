from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import numpy as np
import scipy.io as sio
import cv2
def main():
    parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    img_path = './'  ##要改
    txt_path ='./' ##要改
    txt_name =  ".txt" ##要改
    # txt_name = "test.txt"
    try:
        dice_txt = open(txt_path + txt_name, "r+")
        dice_txt.truncate()  # 如果存在该文件，对先对里面的数据清空
    except IOError:
        dice_txt = open(txt_path + txt_name, "w") #如果不存在该文件，就创建一个
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image liying
    target = np.array([0.99, 0.88, 3])
    for i in range(700):
        image = img_path+str(i+1)+'.jpg'
        result = inference_detector(model, image)
        
        if len(result[0]) !=0:
            for k in range(len(result[0])):
                result[0][k][2]=result[0][k][2]-result[0][k][0]
                result[0][k][3] = result[0][k][3]-result[0][k][1]
                reslist = list(map(int, result[0][k][0:4]))

                confident = result[0][k][4]
                confident = round(confident,3)
                
                if confident == target[1]:
                    confident = '{:.3f}'.format(confident)
                
                if confident == target[0]:
                    print('----------')
                    confident = '{:.3f}'.format(confident)
                dice_txt.write(str(i+1) + ".dat ")
                for j in range(4):
                    dice_txt.write(str(reslist[j]) + " ")
                dice_txt.write(str(confident))
                dice_txt.write("\n")
        else:
            dice_txt.write(str(i + 1) + ".dat 0 0 0 0 0 ")
            dice_txt.write("\n")
    dice_txt.close()



if __name__ == '__main__':
    main()
