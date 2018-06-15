from gmplot import gmplot

lat, lng, rot = 48.013208, 7.833163, 0

# Place map
gmap = gmplot.GoogleMapPlotter(lat, lng, 10)

# Marker
gmap.marker(lat, lng, rot, color = "#FF0000")

# Draw
gmap.draw("6D_GT.html")
