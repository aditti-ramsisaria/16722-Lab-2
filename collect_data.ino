#include <Adafruit_Sensor.h>
#include <DHT.h>

#define DHTPIN 2    // Define the digital pin connected to the DHT22 sensor
#define DHTTYPE DHT22  // Specify the type of DHT sensor you're using

DHT dht(DHTPIN, DHTTYPE);

void setup() {
    Serial.begin(9600);
    dht.begin();
}

void loop() {
    delay(1000);  // Delay between sensor readings

    float temperature = dht.readTemperature();
    float humidity = dht.readHumidity();

    if (isnan(temperature) || isnan(humidity)) {
      Serial.println("Failed to read from DHT sensor");
    } else {
      Serial.print(temperature);
      Serial.print(", ");
      Serial.println(humidity);
    }
}
