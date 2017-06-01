from __future__ import print_function
import os
import simplejson as json
import urllib
import urlparse


def get_white_house_images(url):
    c = urllib.urlopen(url)

    # Get urls of individual images from json response
    j = json.loads(c.read())
    imurls = [im['photo_file_url'] for im in j['photos']]

    # Downlod images
    for url in imurls:
        image = urllib.URLopener()
        image.retrieve(url, os.path.basename(urlparse.urlparse(url).path))
        print('Downloading: ', url)

    return True

if __name__ == '__main__':
    # Query for images
    url = 'http://www.panoramio.com/map/get_panoramas.php?order=popularity&set=public&from=0&to=20&minx=-77.037564&miny=38.896662&maxx=-77.035564&maxy=38.89662&size=medium'

get_white_house_images(url)
