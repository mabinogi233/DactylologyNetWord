import os

if(__name__=='__main__'):
    rootPath = r"C:\Users\LiuWenze\Desktop\ResourceSpace"
    filelist = os.listdir(rootPath)

    for f in filelist:
        filePath1 = os.path.join(rootPath, f)
        if (not os.path.isdir(filePath1)):
            fileName1 = f.split(".")[0]
            # 创建文件夹
            os.mkdir(os.path.join(rootPath, fileName1))
            #读取原路径下的png图片
            file = open(filePath1,'rb')
            content = file.read()
            #写入同名文件夹下 10 次
            for i in range(10):
                file1 = open(os.path.join(rootPath, fileName1,str(i)+'.png'),'wb')
                file1.write(content)
                file1.close()
            file.close()



