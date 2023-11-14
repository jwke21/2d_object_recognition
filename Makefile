# g++ compiler
CXX = /usr/bin/g++

# Compiler flags
CXXFLAGS = -Wall -fdiagnostics-color=always -std=c++17 # 17 for filesystem

# Include paths (including OpenCV Headers)
INCFLAGS = -I/usr/local/include/opencv4/ -Iinclude/

# Library paths
LDFLAGS = -L/usr/local/lib/ -Lobj/

# opencv libraries
LDLIBS = -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio -lopencv_video -lopencv_objdetect

# Directory for object files
OBJDIR = obj

# Directory for final executable
BINDIR = bin

all: imgs obj bin vidDisplay

%.o: src/%.cpp
	$(CXX) $(INCFLAGS) -c $< -o $(OBJDIR)/$@

vidDisplay: csv.o disjoint_set.o util.o vidDisplay.o
	$(CXX) $(CXXFLAGS) $(patsubst %.o,$(OBJDIR)/%.o,$^) -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

imgs:
	mkdir $@

obj:
	mkdir $@

bin:
	mkdir $@

clean:
	rm -f $(OBJDIR)/*.o $(BINDIR)/* *~ 

.PHONY: all clean