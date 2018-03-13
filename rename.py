from os import listdir
from shutil import copyfile

project_path="/home/raunak/Downloads/MIVIA_DB4_dist/extracted_data/testing/"
print(listdir(project_path))


classes=["glass","gunshots","screams"]
no=0
for i in range(1,9):
    folder=project_path+str(i)+"/"
    for j in classes:
        path=folder+j
        print(path)
        dir=listdir(path)
        for k in dir:
            no+=1
            file=path+"/"+k
            print(file)
            a="/home/raunak/Downloads/MIVIA_DB4_dist/final data/testing"
            dst=a+"/"+str(j)+"/"+str(i)+"--"+str(k)
            print(dst)
            print("\n")
            try:
                copyfile(file,dst)
            except FileNotFoundError:
                print(file+" not found")

print(no)