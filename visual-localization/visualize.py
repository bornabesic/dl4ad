from gmplot import gmplot
import utm

zone = (32, "U")
lat, lng = utm.to_latlon(412940.751955, 5318560.37949, *zone)
rot = 0

# Place map
gmap = gmplot.GoogleMapPlotter(lat, lng, 10)

# Marker
gmap.marker(lat, lng, rot, color = "#FF0000")

# Draw
gmap.draw("6D_GT.html")
