CC= g++
CFLAGS = -g -Wall
TARGET = TensorModel

all: $(TARGET)
 
$(TARGET).exe: $(TARGET).o
	$(CC) $(CFLAGS) -o $(TARGET) $(TARGET).o
 
$(TARGET).o: $(TARGET).cpp
	$(CC) $(CFLAGS) -c $(TARGET).cpp

clean:
	rm $(TARGET).o  
