import os
seq_root = "/home/aistudio/data/data35820/sequences"

# sort 之后每个 seq 文件夹中的帧数
n_target_frame = 8
    
# 输出可用图片路径
def output_available_list(n_frame, root):
    # n_frame 为符合数据入口处要求输入的 nFrames，默认为 7
    seq_num = len(os.listdir(root))
    # 以上78为原有数据集的78个文件夹
    output_file = open(os.path.join(seq_root, 'available_list.txt'), mode='w')
    
    for i in range(seq_num):
        seq_dir = os.path.join(root, 'seq_' + str(i+1).zfill(7))
        frame_in_seq = os.listdir(seq_dir)
        frame_in_seq.sort()
        # 不符合训练要求
        if len(frame_in_seq) < n_frame:
            continue
        output_file.write(seq_dir + '\n')
    
        # half = int(n_frame / 2)
        # if n_frame % 2 == 0:
        #     # n_frame 包括目标帧在内为偶数帧，则：前取的帧数与后取的帧数一致
        #     available_frame_path = frame_in_seq[half:-(half-1)]
        # else:
        #     # n_frame 包括目标帧在内为奇数帧
        #     available_frame_path = frame_in_seq[half:-(half-0)]
    
        # 根据上面注释代码缩写
        # 现在连这个也不用了
        '''
        half = int(n_frame / 2)
        available_frame_path = frame_in_seq[half:1- (n_frame % 2) - half]
        for frame_path in available_frame_path:
            output_file.write(os.path.join(seq_dir, frame_path) + '\n')
        '''
            
    
    output_file.close()

output_available_list(n_target_frame, seq_root)
