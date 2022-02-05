# 微信文件整理

整理`Wechat Files/wxid_xxxxxxxxxx/FileStorage/`下的文件

## image: DAT文件解码

`image/`和`Temp`下的文件。

dat文件的加密方式是以一个每个人独特的字节作为key与文件的每个字节进行异或运算。

```shell
python dat_file_decoder.py -d <input_dir> -o <output_dir>
```

参考：https://www.jianshu.com/p/782730f7f016

## video: 不重要视频识别

`video`下的文件，作用是把短视频logo识别出来，分类存放。

```shell
python short_video_filter.py -i <input_dir> -o <output_dir>
```