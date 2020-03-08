import os


# get your current working directory
curDir = os.getcwd()
print(curDir)


# make a new directory 
os.mkdir('newDir')

# rename a directory
os.rename('newDir', 'newDir2')

# To remove a directory
os.rmdir('newDir2')



f= open("guru99.txt","w+")

for i in range(10):
    f.write("This is line %d \r\n" %(i+1))


f.close() 