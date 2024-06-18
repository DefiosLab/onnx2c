#include <M5Core2.h>
#include "types.h"
#include "inference.h"
#include "input.h"

void setup(){
  M5.begin();

  inference(&input,&output);
  float mx = -1000000;
  int label;
  for(int i=0;i<10;i++){
    if(output.data[i] > mx){
      mx = output.data[i];
      label=i;
    }
  }
  char buffer[50];
  sprintf(buffer, "label:%d\n", label);
  M5.Lcd.setTextSize(4); 
  M5.Lcd.print(buffer);
}

void loop() {

}
