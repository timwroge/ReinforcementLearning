int data;

void setup() { 
  Serial.begin(9600);                               //initialize serial COM at 9600 baudrate
  pinMode(LED_BUILTIN, OUTPUT);                    //declare the LED pin (13) as output
  digitalWrite (LED_BUILTIN, LOW);                     //Turn OFF the Led in the beginning
  
  Serial.println("Hello! Connection Successful.");
}
 
void loop() {
while (Serial.available())    //whatever the data that is coming in serially and assigning the value to the variable “data”

{ 

data = Serial.read();

}

if (data == '1')

digitalWrite (LED_BUILTIN, HIGH);                  //Turn On the Led

else if (data == '0')

digitalWrite (LED_BUILTIN, LOW);                  //Turn OFF the Led

}
