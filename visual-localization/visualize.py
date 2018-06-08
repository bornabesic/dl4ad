from gmplot import gmplot

# Place map
gmap = gmplot.GoogleMapPlotter.from_geocode("Technische Fakultät, Freiburg")

# Marker
lat, lon, rot = 48.013208, 7.833163, -20
gmap.marker(lat, lon, rot, color = "#FF0000")

# Draw
gmap.draw("6D_GT.html")

