

#                           clear "#" start lines       



file_name="evaluation.py"

with open(file_name,"r",encoding="utf-8")as f:
    lines=f.readlines()

lines=[l for l in lines if not l.lstrip().startswith("#")]    

with open(file_name,"w",encoding="utf-8")as f:
     f.writelines(lines)