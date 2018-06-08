from gmplot import gmplot

# Place map
gmap = gmplot.GoogleMapPlotter(37.766956, -122.438481, 13)

# Marker
lat, lon, rot = 37.770776, -122.461689, -20
gmap.marker(lat, lon, rot, color = "#FF0000")

# Draw
gmap.draw("6D_GT.html")

