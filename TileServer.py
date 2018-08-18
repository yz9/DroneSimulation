import os
from PIL import Image
from PIL import ImageDraw
import random
import urllib
import math

# Bing Maps Tile Template:
# http://{SERVER}/tiles/a{QUAD_KEY}.jpeg?token={TOKEN}&mkt={LOCALE}&g={??}

class TileServer(object):
    def __init__(self):
        self.imdict = {}
        self.surfdict = {}
        self.layers = 'SATELLITE'
        self.path = './'
        self.tilesPath = self.path + 'tiles/'
        if not os.path.exists(self.tilesPath):
            os.makedirs(self.tilesPath)
        self.urltemplate = 'http://ecn.t{4}.tiles.virtualearth.net/tiles/{3}{5}?g=0'
        self.layerdict = {'SATELLITE': 'a', 'HYBRID': 'h', 'ROADMAP': 'r'}

        self.EarthRadius = 6378137
        self.MinLatitude = -85.05112878
        self.MaxLatitude = 85.05112878
        self.MinLongitude = -180
        self.MaxLongitude = 180


    def tiletoquadkey(self, x, y, z):
        quadKey = ''
        for i in range(z, 0, -1):
            digit = 0
            mask = 1 << (i - 1)
            if(x & mask) != 0:
                digit += 1
            if(y & mask) != 0:
                digit += 2
            quadKey += str(digit)
        return quadKey

    def loadimage(self, fullname, tilekey):
        im = Image.open(fullname)
        self.imdict[tilekey] = im
        return self.imdict[tilekey]


    def tile_as_image(self, xi, yi, zoom):
        tilekey = (xi, yi, zoom)
        result = None
        try:
            result = self.imdict[tilekey]
        except:
            filename = '{}_{}_{}_{}.jpg'.format(zoom, xi, yi, self.layerdict[self.layers])
            fullname = self.tilesPath + filename
            try:
                result = self.loadimage(fullname, tilekey)
            except:
                server = random.choice(range(1,4))
                quadkey = self.tiletoquadkey(*tilekey)
                #print(quadkey)
                #print(server)
                #print(self.layerdict[self.layers])
                url = self.urltemplate.format(xi, yi, zoom, self.layerdict[self.layers], server, quadkey)
                #print(url)
                #print(filename)
                print("Downloading map tile %s to local cache." % filename)

                # url = "http://shop.yoshimura-jp.com/files/img1/1363245369_cd7b755dfba76f550c7983d4f1833701.jpg"
                urllib.urlretrieve(url, fullname)
                result = self.loadimage(fullname, tilekey)
        return result

    def tile_as_image_from_corr(self, latitude, longitude, zoomLevel):
        sinLatitude = math.sin(latitude * math.pi/180)
        pixelX = ((longitude + 180) / 360) * 256 * 2 ** zoomLevel
        pixelY = (0.5 - math.log((1 + sinLatitude) / (1 - sinLatitude)) / (4 * math.pi)) * 256 * 2 ** zoomLevel
        tileX = math.floor(pixelX/256.0)
        tileY = math.floor(pixelY/256.0)
        return self.tile_as_image(int(tileX), int(tileY), zoomLevel)

    def tiles_as_image_from_corr(self, latitude, longitude, zoomLevel, tilesX, tilesY, tilesOffsetX, tilesOffsetY):
        """ Return a large image with top-left at the specified lat/lon comprised of the desired set of
        tiles.  Use the cache if available, otherwise download the required data.
        Uses EPSG:3857 -- WGS84 Web Mercator
        """
        sinLatitude = math.sin(latitude * math.pi/180)
        pixelX = ((longitude + 180) / 360) * 256 * 2 ** zoomLevel
        pixelY = (0.5 - math.log((1 + sinLatitude) / (1 - sinLatitude)) / (4 * math.pi)) * 256 * 2 ** zoomLevel

        # print "Input LL:", latitude, longitude

        # offset tiles
        pixelX = pixelX-256*tilesOffsetX
        pixelY = pixelY-256*tilesOffsetY


        tileX = math.floor(pixelX/256.0)
        tileY = math.floor(pixelY/256.0)

        actual_pX = tileX*256;
        actual_pY = tileY*256;
        actual_lat, actual_lon = self.PixelXYToLatLong(actual_pX, actual_pY, zoomLevel)

        # print "Actual pixel:", actual_pX, actual_pY
        # print "Actual LL:", actual_lat, actual_lon

        sizeX = 256*tilesX
        sizeY = 256*tilesY
        im = Image.new("RGB", (sizeX, sizeY), "white")
        tempImage = None
        for xt in range(0,tilesX,1):
            for yt in range(0,tilesY,1):
                tempImage = self.tile_as_image(int(tileX)+xt, int(tileY)+yt, zoomLevel)
                # draw edges around each tile by filling a slightly expanded square. -gd
                draw = ImageDraw.Draw(im)
                thickness = 8
                draw.rectangle(((xt*256-thickness, yt*256-thickness), (xt*256+256+thickness, yt*256+256+thickness)), fill="red")
                # put in image
                im.paste(tempImage, (xt*256, yt*256))


        filename = '{}_{}_{}_{}_{}_{}_{}.png'.format(tileX, tileY, zoomLevel, tilesX, tilesY, tilesOffsetX, tilesOffsetY)
        fullname = self.tilesPath + filename
        im.save(fullname, "PNG")
        return im, actual_pX, actual_pY, fullname

    def imagePixelsToLL(self, actual_pX, actual_pY, zoomLevel, imx, imy ):
        """
        Convert image pixels from a EPSG:4326 / WGS84  tile image coordinate to lat,long. These are not the WGS 
        "Earth Pixels" used elsewhere in this module.

        Must be called with (actual_pX, actual_pY) which is earth position of the top-left corner of the 
        picture returned by tiles_as_image_from_corr (second and third paramaters).
        """
        tileX = actual_pX/256+imx
        recoveredPixelX = tileX*256

        tileY = actual_pY/256+imy
        recoveredPixelY = tileY*256
        recoveredLatitude = math.asin(math.tanh((4 * math.pi) * (-(recoveredPixelY/( 256 * 2 ** zoomLevel)-0.5))))

        return self.PixelXYToLatLong(recoveredPixelX, recoveredPixelY, zoomLevel)

    # /// <summary>
    # /// Clips a number to the specified minimum and maximum values.
    # /// </summary>
    # /// <param name="n">The number to clip.</param>
    # /// <param name="minValue">Minimum allowable value.</param>
    # /// <param name="maxValue">Maximum allowable value.</param>
    # /// <returns>The clipped value.</returns>
    def Clip(self, n, minValue, maxValue):
        if (n< minValue):
            return minValue
        if (n > maxValue):
            return maxValue
        return n

    # /// <summary>
    # /// Determines the map width and height (in pixels) at a specified level
    # /// of detail.
    # /// </summary>
    # /// <param name="levelOfDetail">Level of detail, from 1 (lowest detail)
    # /// to 23 (highest detail).</param>
    # /// <returns>The map width and height in pixels.</returns>
    def MapSize(self, levelOfDetail):
        return 256 << levelOfDetail

    # /// <summary>
    # /// Determines the ground resolution (in meters per pixel) at a specified
    # /// latitude and level of detail.
    # /// </summary>
    # /// <param name="latitude">Latitude (in degrees) at which to measure the
    # /// ground resolution.</param>
    # /// <param name="levelOfDetail">Level of detail, from 1 (lowest detail)
    # /// to 23 (highest detail).</param>
    # /// <returns>The ground resolution, in meters per pixel.</returns>
    def GroundResolution(self, latitude, levelOfDetail):
        latitude = self.Clip(latitude, self.MinLatitude, self.MaxLatitude)
        return math.cos(latitude * math.pi / 180) * 2 * math.pi * self.EarthRadius / self.MapSize(levelOfDetail);

    # /// <summary>
    # /// Converts a pixel from pixel XY coordinates at a specified level of detail
    # /// into latitude/longitude WGS-84 coordinates (in degrees).
    # /// These pixels are Earth coordinates in the EPSG:3857 WGS 84 / Pseudo-Mercator system.
    # /// </summary>
    # /// <param name="pixelX">X coordinate of the point, in pixels.</param>
    # /// <param name="pixelY">Y coordinates of the point, in pixels.</param>
    # /// <param name="levelOfDetail">Level of detail, from 1 (lowest detail)
    # /// to 23 (highest detail).</param>
    # /// <param name="latitude">Output parameter receiving the latitude in degrees.</param>
    # /// <param name="longitude">Output parameter receiving the longitude in degrees.</param>
    #
    # Web testing interface
    # See: https://epsg.io/transform#s_srs=3857&t_srs=4326
    def PixelXYToLatLong(self, pixelX, pixelY, levelOfDetail):
        # Would expect a basic inverse like:
        # recoveredLatitude = math.asin(math.tanh((4 * math.pi) * (-(recoveredPixelY/( 256 * 2 ** zoomLevel)-0.5))))
        # but instead it's this...
        #
        mapSize = self.MapSize(levelOfDetail)
        x = (self.Clip(pixelX, 0, mapSize - 1) / mapSize) - 0.5
        y = 0.5 - (self.Clip(pixelY, 0, mapSize - 1) / mapSize)
        latitude = 90 - 360 * math.atan(math.exp(-y * 2 * math.pi)) / math.pi
        longitude = 360 * x
        return latitude, longitude

    #
    # Web testing interface
    # See: https://epsg.io/transform#s_srs=3857&t_srs=4326
    def LatLongToPixelXY(self, latitude, longitude, zoomLevel):
        sinLatitude = math.sin(latitude * math.pi/180)
        # These pixels are Earth coordinates in the EPSG:4326 system
        # See: https://en.wikipedia.org/wiki/Web_Mercator
        pixelX = ((longitude + 180) / 360) * 256 * 2 ** zoomLevel
        pixelY = (0.5 - math.log((1 + sinLatitude) / (1 - sinLatitude)) / (4 * math.pi)) * 256 * 2 ** zoomLevel
        return pixelX, pixelY


if __name__ == "__main__":
    lat, lon = 45.505955, -73.576675
    lat, lon = 45.554925, -73.701590 # Boulevard Cartier O, Laval, H7N
    lat, lon = 46.26569, -74.133841  # lac ouareau lanaudiere
    lat, lon = 45.4341, -73.59532    # verdun
    lat, lon = 45.44203, -73.602995    # verdun
    zoomLevel = 18

    ts = TileServer()
    tilesX = 22
    tilesY = 22
    tilesOffsetX = 0
    tilesOffsetY = 0
    im, actual_pX, actual_pY, filename = ts.tiles_as_image_from_corr(lat, lon, zoomLevel, tilesX, tilesY, tilesOffsetX,
                                                           tilesOffsetY)
    im.show()
