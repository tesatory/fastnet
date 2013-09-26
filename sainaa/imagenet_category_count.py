import xml.etree.ElementTree as ET
import os.path

def read_synset(e, depth, out):
    count = 0
    if os.path.exists(data_dir + e.attrib['wnid']):
        count += 1
    
    for c in e.getchildren():
        count += read_synset(c, depth+1, out)

    if count > 50:
        out.append(('  '*depth) +' '+ str(count) +' '+ e.attrib['words'])

    return count

data_dir = '/scratch/sainaa/imagenet/train/'
tree = ET.parse('./ImageNetToolboxV0.3/structure_released.xml')
root = tree.getroot()
out = list()
read_synset(root.getchildren()[1], 0, out)
out.reverse()
for l in out:
    print l
