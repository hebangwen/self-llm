import h5py
import numpy as np
import argparse

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="从HDF5文件中提取前N条数据并保存为新的HDF5文件")
    parser.add_argument("input_file", type=str, help="输入的HDF5文件路径（如 train.h5）")
    parser.add_argument("num_samples", type=int, help="要提取的样本数量 N")
    parser.add_argument("output_file", type=str, help="输出的HDF5文件路径（如 train_subset.h5）")

    args = parser.parse_args()

    input_path = args.input_file
    num_samples = args.num_samples
    output_path = args.output_file

    print(f"读取文件: {input_path}")
    print(f"提取前 {num_samples} 条数据")
    print(f"保存到文件: {output_path}")

    with h5py.File(input_path, "r") as src_file:
        # 获取 waveforms 和 labels 数据集
        if 'waveforms' not in src_file or 'labels' not in src_file:
            raise KeyError("HDF5 文件中缺少 'waveforms' 或 'labels' 数据集")

        total_samples = src_file['waveforms'].shape[0]
        if num_samples > total_samples:
            raise ValueError(f"请求的样本数 {num_samples} 超出文件总样本数 {total_samples}")

        # 读取前 N 条数据
        waveforms = src_file['waveforms'][:num_samples]
        labels = src_file['labels'][:num_samples]

        print(f"waveforms shape: {waveforms.shape}")
        print(f"labels shape: {labels.shape}")

        # 写入新文件
        with h5py.File(output_path, "w") as dst_file:
            # 创建数据集
            dst_file.create_dataset("waveforms", data=waveforms, dtype=np.complex64)
            dst_file.create_dataset("labels", data=labels)

            print(f"成功保存前 {num_samples} 条数据到 {output_path}")

if __name__ == "__main__":
    main()

