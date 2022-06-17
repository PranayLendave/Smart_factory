#include <WiFi.h>

const char* ssid = "change_your_ssid";
const char* password =  "change_your_passward";

WiFiServer wifiServer(5050);
const byte led_red = 12;
const byte led_green = 13;
const byte led_blue = 15;
char c=x;
void setup() {


  Serial.begin(115200);

  delay(1000);

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi..");
  }

  Serial.println("Connected to the WiFi network");
  Serial.println(WiFi.localIP());

  wifiServer.begin();
  pinMode(led_red, OUTPUT);
  pinMode(led_green, OUTPUT);
  pinMode(led_blue, OUTPUT);
  digitalWrite(led_red, HIGH);
  digitalWrite(led_blue, HIGH);
  digitalWrite(led_green, HIGH);
  delay(1000);
}

void loop() {

  WiFiClient client_1 = wifiServer.available();

  if (client_1) {

    while (client_1.connected()) {
      //Serial.println("[Connected] client connected");

      while (client_1.available() > 0) {
        // Serial.println("Listening");
        c = client_1.read();
        Serial.print("Read form server: ");
        Serial.print(c);
        Serial.println();
        client_1.write(c);
      }

      delay(10);
    }

    client_1.stop();
    Serial.println("Client disconnected");

  }  if (c == 'a') {
    Serial.println("Empty jar");
    digitalWrite(led_red, LOW);
    digitalWrite(led_green, HIGH);
    digitalWrite(led_blue, HIGH);
    delay(1000);
  }
  else if (c == 'b') {
    Serial.println("no_cap");
    digitalWrite(led_red, HIGH);
    digitalWrite(led_green, LOW);
    digitalWrite(led_blue, HIGH);
    delay(1000);
  }
  else if (c == 'c') {
    Serial.println("no_label");
    digitalWrite(led_red, HIGH);
    digitalWrite(led_green, HIGH);
    digitalWrite(led_blue, LOW);
    delay(1000);
  }
  else if (c == 'e') {
    Serial.println("Perfect jar");
    digitalWrite(led_red, LOW);
    digitalWrite(led_green, LOW);
    digitalWrite(led_blue, LOW);
    delay(1000);
  }
  else {
    digitalWrite(led_red, HIGH);
    digitalWrite(led_green, HIGH);
    digitalWrite(led_blue, HIGH);
    delay(1000);
  }
}
