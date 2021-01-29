import  hyperlpr as pp
import cv2
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import numpy as np
import os

fontC = ImageFont.truetype("Font/platech.ttf", 30)
def draw_info(image, rect, addText):

#    cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int( rect[2]), int(rect[3])), (0, 0, 255), 1,cv2.LINE_AA)
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.text((int(rect[0]), int(rect[1] - 35)), addText, (0, 0, 255), font=fontC)
    imagex = np.array(img)
    return imagex

res_path = r'output'
os.makedirs(res_path, exist_ok=True)
def main(recogImg):
   # img = cv2.imread(recogImg)
    img = cv2.imdecode(np.fromfile(recogImg, dtype=np.uint8), cv2.IMREAD_COLOR)
    res = pp.HyperLPR_plate_recognition(img)
    if not res:
        print("未识别")
        return
    showImage=draw_info(img,res[0][2],res[0][0])
#    cv2.imwrite(os.path.join(res_path, 'demo_result.jpg'), showImage)
    #cv2.imencode(res_path, showImage)[1].tofile(res_path)
#    cv2.imshow("showImage",showImage)
 #   cv2.waitKey(5000)

if __name__ == '__main__':
    main("demo_detec.jpg ")
    print("预识别完成，结果保存在output文件夹中")



