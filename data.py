import csv

gender_mapping = {'':2, 'f':1, 'm':0, 'u':2}

def list_images(path, base_path, prefix):
    """
    Get all the images and labels in directory/label/*.jpg
    """
    filenames=[]
    labels=[]
    
    with open(path) as fp:
        next(fp) #skip csv header
        for row in csv.reader(fp, delimiter="\t"):
            fn=row[1]
            uid=row[0]
            gender=gender_mapping[row[4]]
            x=row[2]+"."#"1."
            
            filenames.append(base_path+uid+"/"+prefix+x+fn)
            labels.append(gender)
            
    return filenames, labels