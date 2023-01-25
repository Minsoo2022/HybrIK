import os
import shutil
import glob
# print(1)
# os.listdir('../../../data/3dhp/mpi_inf_3dhp_train_set')
# qq = '../../../data/3dhp/mpi_inf_3dhp_train_set/S2/Seq1/video_7/002588.jpg'
# 'S8/Seq1/images/S8_Seq1_V1/img_S8_Seq1_V1_003711.jpg'
# qq[0].split('/')[6:]
#


image_list = glob.glob('../../../data/3dhp/mpi_inf_3dhp_train_set/*/*/*/*.jpg')

for img in image_list:
    q = img.split('/')
    video_v = q[8].split('_')[1]
    os.makedirs(f'../../../data/3dhp_processing/{q[6]}/{q[7]}/images/{q[6]}_{q[7]}_V{video_v}', exist_ok=True)
    print(f'{q[6]}/{q[7]}/images/{q[6]}_{q[7]}_V{video_v}/img_{q[6]}_{q[7]}_V{video_v}_{q[9]}')
    shutil.move(img, f'../../../data/3dhp_processing/{q[6]}/{q[7]}/images/{q[6]}_{q[7]}_V{video_v}/img_{q[6]}_{q[7]}_V{video_v}_{q[9]}')
