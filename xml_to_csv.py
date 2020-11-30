import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    i=0
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):


            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     float(member[1][0].text),
                     float(member[1][1].text),
                     float(member[1][2].text),
                     float(member[1][3].text)
                     )
            print(value)
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

##'train'
def main():
    for folder in ['trainarm']:
        image_path = os.path.join(os.getcwd(), ('annotations_voc_all/' + folder))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(('annotations_voc_all/' + folder + '_labels.csv'), index=None)
        print('Successfully converted xml to csv.')


main()
