import zipfile
 
f = zipfile.ZipFile("./data.zip",'r') # 压缩文件位置
for file in f.namelist():
    f.extract(file,"./datasets/")               # 解压位置
f.close()